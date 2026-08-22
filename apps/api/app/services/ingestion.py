from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import unicodedata
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.utils import source_type_from_path
from app.models import (
    Chunk,
    ChunkCoordinate,
    ChunkRelationGraphState,
    ChunkSpan,
    ChunkStructureEdge,
    ChunkStructureMapping,
    ChunkStructureNode,
    ChunkVersion,
    CoarseConceptState,
    ContextGraphState,
    Document,
    DocumentVersion,
    IngestionBatch,
    IngestionBatchRecovery,
    IngestionCompensationLog,
    IngestionFileStage,
    IngestionJob,
    KnowledgeBaseVectorRuntimeState,
    KnowledgeBase,
    MidConceptState,
    ParseJob,
    SourceFile,
    VectorRecord,
)
from app.services.cancellation import IngestionCancelled, ensure_not_cancelled
from app.services.context_graph import (
    CONTEXTUAL_INDEX_REPAIR_MODE_VERIFY_ONLY,
    VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION,
    active_chunks_query,
    build_local_context_hints,
    concept_provider_output_failure_card,
    compute_chunk_scope_hash,
    context_graph_overview_stats,
    context_graph_stats,
    verify_contextual_indexes_for_graph_only_retry,
    invalidate_context_graph_cache_after_commit,
    mark_context_graph_active_scope_stale,
    rebuild_context_graph,
    rewrite_chunk_version_state_for_active_scope_mutation,
    write_chunks_and_structure,
    write_contextual_indexes,
)
from app.services.embeddings import is_degraded_mode
from app.services.error_sanitizer import (
    external_failure_classification,
    public_exception_message,
)
from app.services.ingestion_logs import emit_ingestion_log
from app.services.ingestion_resource_lock import (
    INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
    IngestionResourceBusyError,
    active_ingestion_resource_lease,
    advisory_lock_key,
    ingestion_batch_recovery_fence_diagnostics,
    ingestion_batch_recovery_owner_token_for_batch,
    ingestion_resource_lock_wait_diagnostics,
    knowledge_base_ingestion_resource_lock,
    knowledge_base_resource_key,
)
from app.services.parsers import derive_partition, parse_document, sections_to_json
from app.services.language_metadata import (
    apply_language_identity,
    detect_document_language,
    explicit_language_from_pending_metadata,
    language_identity_from_record,
    language_identity_summary,
    normalize_explicit_language_tag,
    pending_language_record_fields,
)
from app.services import runtime_settings as runtime_settings_service
from app.services.runtime_settings import (
    refresh_runtime_settings_if_needed,
)
from app.services.storage import (
    ALLOWED_UPLOAD_SUFFIXES,
    LOGICAL_SOURCE_SLOT_PROTOCOL_VERSION,
    SOURCE_SNAPSHOT_PROTOCOL_VERSION,
    UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION,
    SourceSnapshotError,
    SourceSnapshotNotFoundError,
    UploadChecksumMismatchError,
    UploadValidationError,
    VerifiedSourceIdentity,
    contained_path,
    compute_checksum,
    durable_unlink,
    durable_ensure_directory,
    durable_sync_directory,
    ensure_knowledge_base_storage_durability_ready,
    ensure_storage_durability_ready,
    freeze_existing_source_snapshot,
    freeze_verified_readonly_import_root,
    namespace_durability_protocol,
    normalize_upload_filename,
    normalize_upload_source_slot_key,
    open_verified_readonly_import_file,
    require_storage_durability_capability,
    replay_frozen_source_snapshot,
    run_bounded_source_io,
    snapshot_source_file,
    source_slot_key_for_path,
    source_snapshot_path,
    validate_knowledge_base_source_path,
    validate_source_content_path,
    verified_readonly_import_checksum,
    verified_source_checksum,
)
from app.services.qdrant_outbox import (
    QDRANT_DELETE_SOURCE_BINDING_PROTOCOL_VERSION,
    QDRANT_INGESTION_PARSE_COMPENSATION_REASON,
    QDRANT_OUTBOX_PROTOCOL_VERSION,
    find_qdrant_delete_intent_by_source_binding,
    mark_qdrant_delete_attempts_committed,
    persist_qdrant_delete_attempt,
    replay_qdrant_delete_intent,
    validated_committed_qdrant_upsert_target,
    validated_qdrant_delete_intent,
)
from app.services.vector_store import VectorStore


ALLOWED_SUFFIXES = ALLOWED_UPLOAD_SUFFIXES
EXCLUDED_PARTS = {"output", "scripts", ".ipynb_checkpoints", "__pycache__"}
IGNORED_NAMES = {".ds_store"}
RAW_SOURCE_MANIFEST_FILENAME = "raw-manifest.json"
RAW_SOURCE_MANIFEST_PROTOCOL_VERSION = "symbograph_raw_source_manifest_v1"
RAW_SOURCE_MANIFEST_MAX_FILES = 100_000
RAW_SOURCE_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
RAW_IMPORT_FORBIDDEN_PARTS = frozenset(
    {
        ".pending",
        "ingestion",
        "source_snapshots",
        "source_slots",
    }
)
TERMINAL_STATES = {"completed", "failed", "partial_failed", "skipped", "cancelled", "cancel_failed"}
ACTIVE_FILE_JOB_STATES = {"queued", "parsing", "chunking", "embedding", "extracting_graph", "processing"}
CANCELLING_STATES = {"cancel_requested", "cancelling", "compensating"}
CANCEL_TERMINATE_SIGNAL = "SIGTERM"
CELERY_TERMINAL_TASK_STATES = frozenset({"SUCCESS", "FAILURE", "REVOKED"})
DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION = "document_metadata_intent_v2"
DOCUMENT_METADATA_RELEASE_RECOVERY_PROTOCOL_VERSION = "document_metadata_release_recovery_v1"
DOCUMENT_METADATA_INTENT_KEY = "document_metadata_intent"
INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION = "ingestion_batch_cancel_compensation_v1"
INGESTION_BATCH_RECOVERY_BEFORE_PROTOCOL_VERSION = "ingestion_batch_before_scope_v1"
INGESTION_FILE_BEFORE_PROTOCOL_VERSION = "ingestion_file_before_scope_v1"
INGESTION_FILE_WRITE_SET_PROTOCOL_VERSION = "ingestion_file_committed_write_set_v1"
INGESTION_GRAPH_WRITE_SET_PROTOCOL_VERSION = "ingestion_graph_committed_write_set_v1"
VERSIONED_GRAPH_RETRY_PROTOCOL_VERSION = "versioned_graph_retry_v1"
INGESTION_RECOVERY_PARSE_UNCOMMITTED_STATUSES = frozenset(
    {
        "prepared",
        "parsing",
        "parse_compensation_pending",
        "parse_compensating",
        "parse_compensated",
        "completed_no_writes",
        "manual_review",
    }
)
INGESTION_RECOVERY_PARSE_COMMITTED_STATUSES = frozenset(
    {
        "graph_building",
        "graph_compensation_pending",
        "graph_compensated",
        "completed",
        "manual_review",
    }
)
INGESTION_FILE_STAGE_STATUS_PHASES = {
    "prepared": "prepared",
    "parsing": "parsing",
    "indexed_committed": "indexed",
    "failed": "failed",
    "cancel_observed": "cancel_observed",
    "compensation_pending": "qdrant_compensation",
    "compensating": "database_restore",
    "compensated": "compensated",
    "retained_after_parse_commit": "context_graph",
    "manual_review": "manual_review",
}
DOCUMENT_METADATA_FIELDS = (
    "title",
    "source_path",
    "source_type",
    "language",
    "language_source",
    "language_confidence",
    "language_detection_protocol_version",
    "language_detection_hash",
    "language_metadata_json",
    "tags",
    "difficulty",
    "checksum",
    "visibility",
    "is_active",
)
DOCUMENT_METADATA_BEFORE_STATE_FIELDS = (
    "exists",
    "document_id",
    "metadata",
    "created_at",
    "updated_at",
    "active_document_version_ids",
    "active_chunk_ids",
    "knowledge_base_chunk_version",
)
DOCUMENT_METADATA_CANDIDATE_STATE_FIELDS = ("metadata",)
DOCUMENT_METADATA_PENDING_INTENT_FIELDS = (
    "protocol_version",
    "schema_hash",
    "intent_id",
    "knowledge_base_id",
    "source_path",
    "status",
    "created_at",
    "document_id",
    "before_state",
    "before_state_hash",
    "candidate_state",
    "candidate_state_hash",
    "candidate_revision",
    "candidate_history",
    "pending_payload_hash",
)
DOCUMENT_METADATA_CANDIDATE_HISTORY_FIELDS = ("revision", "candidate_state_hash", "replaced_at")
SOURCE_FILE_DELETE_OPERATION = "source_file_delete"
SOURCE_FILE_DELETE_LOCK_OPERATION = "remove_knowledge_base_file"
SOURCE_FILE_DELETE_PROTOCOL_VERSION = "source_file_delete_v1"
SOURCE_FILE_DELETE_NAMESPACE_DURABILITY_PROTOCOLS = frozenset(
    {
        "posix_parent_directory_fsync_v1",
        "windows_volume_flush_v1",
        "windows_pytest_adapter_v1",
    }
)
SOURCE_FILE_DELETE_ACTIVE_STATUSES = frozenset(
    {
        "pending",
        "external_failed",
        "external_applied",
        "cache_invalidation_pending",
        "manual_review",
    }
)
SOURCE_FILE_DELETE_STATUS_PHASES = {
    "pending": frozenset({"intent_committed"}),
    "external_failed": frozenset({"intent_committed"}),
    "external_applied": frozenset({"external_applied"}),
    "cache_invalidation_pending": frozenset({"database_committed"}),
    "manual_review": frozenset(
        {"intent_committed", "external_applied", "database_committed"}
    ),
    "committed": frozenset({"completed"}),
}
SOURCE_FILE_DELETE_PAYLOAD_FIELDS = (
    "protocol_version",
    "namespace_durability_protocol",
    "schema_hash",
    "intent_id",
    "knowledge_base_id",
    "knowledge_base_name",
    "source_path",
    "phase",
    "file_before",
    "document_before",
    "source_files_before",
    "database_commit",
    "history",
    "created_at",
    "updated_at",
    "intent_payload_hash",
)
SOURCE_FILE_DELETE_FILE_STATE_FIELDS = ("exists", "checksum", "size_bytes")
SOURCE_FILE_DELETE_SOURCE_FILE_CARD_FIELDS = (
    "source_file_id",
    "document_id",
    "source_path",
    "checksum",
    "source_type",
    "size_bytes",
    "updated_at",
)
SOURCE_FILE_DELETE_DOCUMENT_STATE_FIELDS = (
    "exists",
    "document_id",
    "source_path",
    "checksum",
    "is_active",
    "updated_at",
    "active_chunk_count",
    "active_chunk_scope_hash",
)
SOURCE_FILE_DELETE_DATABASE_COMMIT_FIELDS = (
    "committed_at",
    "deleted_chunk_count",
    "deleted_chunk_ids_sample",
    "deleted_chunk_ids_truncated_count",
    "deleted_source_file_count",
    "active_chunk_scope_hash_after",
    "knowledge_base_chunk_version_after",
    "chunk_version_state_id",
    "chunk_version_state_hash",
    "context_graph_state_id",
    "freshness_rows_marked_stale",
    "cache_invalidation_required",
    "cache_invalidation_dispatched_at",
    "qdrant_cleanup_performed",
)
SOURCE_FILE_DELETE_HISTORY_FIELDS = ("from", "to", "at", "details")


class DocumentMetadataRestoreError(RuntimeError):
    """Raised when a failed ingest cannot prove restoration of PostgreSQL facts."""


class SourceFileDeleteError(RuntimeError):
    """Base error for the durable single-source deletion protocol."""


class SourceFileDeleteConflict(SourceFileDeleteError):
    """Raised when a deletion cannot safely select or fence its source slot."""


class SourceFileDeleteIntegrityError(SourceFileDeleteError):
    """Raised when a persisted delete intent or source identity is untrusted."""


class SourceFileDeleteRecoveryError(SourceFileDeleteError):
    """Raised after a recoverable deletion failure has been durably recorded."""


def _canonical_payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def document_metadata_intent_schema_hash() -> str:
    return _canonical_payload_hash(
        {
            "protocol_version": DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION,
            "pending_intent_fields": list(DOCUMENT_METADATA_PENDING_INTENT_FIELDS),
            "before_state_fields": list(DOCUMENT_METADATA_BEFORE_STATE_FIELDS),
            "candidate_state_fields": list(DOCUMENT_METADATA_CANDIDATE_STATE_FIELDS),
            "metadata_fields": list(DOCUMENT_METADATA_FIELDS),
            "candidate_history_fields": list(DOCUMENT_METADATA_CANDIDATE_HISTORY_FIELDS),
            "extra_fields": "forbid",
        }
    )


def _pending_intent_payload_hash(intent: dict[str, Any]) -> str:
    return _canonical_payload_hash({key: value for key, value in intent.items() if key != "pending_payload_hash"})


def _refresh_pending_intent_payload_hash(intent: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(intent)
    refreshed["pending_payload_hash"] = _pending_intent_payload_hash(refreshed)
    return refreshed


def _validate_document_metadata_schema(metadata: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(metadata, dict) or set(metadata) != set(DOCUMENT_METADATA_FIELDS):
        raise DocumentMetadataRestoreError(f"{label} has an invalid metadata schema")
    if not all(isinstance(metadata[field], str) and metadata[field] for field in ("title", "source_path", "source_type", "checksum", "visibility")):
        raise DocumentMetadataRestoreError(f"{label} has invalid required string metadata")
    if metadata["language"] is not None and not isinstance(metadata["language"], str):
        raise DocumentMetadataRestoreError(f"{label} language must be a string or null")
    if metadata["language_source"] is not None and metadata["language_source"] not in {
        "explicit_metadata",
        "deterministic_detection",
        "unknown",
    }:
        raise DocumentMetadataRestoreError(f"{label} language_source is invalid")
    confidence = metadata["language_confidence"]
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise DocumentMetadataRestoreError(
            f"{label} language_confidence must be a number from 0 to 1 or null"
        )
    for field in (
        "language_detection_protocol_version",
        "language_detection_hash",
    ):
        if metadata[field] is not None and not isinstance(metadata[field], str):
            raise DocumentMetadataRestoreError(f"{label} {field} must be a string or null")
    if not isinstance(metadata["language_metadata_json"], dict):
        raise DocumentMetadataRestoreError(f"{label} language_metadata_json must be an object")
    if metadata["difficulty"] is not None and not isinstance(metadata["difficulty"], str):
        raise DocumentMetadataRestoreError(f"{label} difficulty must be a string or null")
    if not isinstance(metadata["tags"], list) or not all(isinstance(item, str) for item in metadata["tags"]):
        raise DocumentMetadataRestoreError(f"{label} tags must be a string list")
    if not isinstance(metadata["is_active"], bool):
        raise DocumentMetadataRestoreError(f"{label} is_active must be boolean")
    return dict(metadata)


def _document_metadata_payload(document: Document) -> dict[str, Any]:
    return {
        "title": document.title,
        "source_path": document.source_path,
        "source_type": document.source_type,
        "language": document.language,
        "language_source": document.language_source,
        "language_confidence": document.language_confidence,
        "language_detection_protocol_version": document.language_detection_protocol_version,
        "language_detection_hash": document.language_detection_hash,
        "language_metadata_json": dict(document.language_metadata_json or {}),
        "tags": list(document.tags or []),
        "difficulty": document.difficulty,
        "checksum": document.checksum,
        "visibility": document.visibility,
        "is_active": bool(document.is_active),
    }


def _is_storage_identity_title(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _document_display_title(
    path: Path,
    *,
    document: Document | None = None,
    display_filename: str | None = None,
    display_title: str | None = None,
    logical_source_slot_key: str | None = None,
    source_slot_protocol_version: str | None = None,
) -> str:
    """Keep physical content-addressed paths out of user-visible titles.

    Upload storage is deliberately checksum-addressed.  A reparse therefore
    sees a digest filename even though the logical upload slot still owns the
    validated original filename.  Preserve the exact existing title when it
    is usable, and recover legacy rows from the logical slot when an older
    reparse already replaced the title with the digest.
    """

    if display_filename:
        return Path(display_filename).stem or Path(display_filename).name
    if display_title:
        normalized_display_title = str(display_title).strip()
        if normalized_display_title and not _is_storage_identity_title(normalized_display_title):
            return normalized_display_title[:255]

    existing_title = str(document.title or "").strip() if document is not None else ""
    if existing_title and not _is_storage_identity_title(existing_title):
        return existing_title

    effective_slot_key = (
        logical_source_slot_key
        or (document.logical_source_slot_key if document is not None else None)
    )
    effective_slot_protocol = (
        source_slot_protocol_version
        or (document.source_slot_protocol_version if document is not None else None)
    )
    if (
        effective_slot_protocol == UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION
        and isinstance(effective_slot_key, str)
        and effective_slot_key.startswith("upload/")
    ):
        logical_filename = effective_slot_key.removeprefix("upload/")
        recovered_title = Path(logical_filename).stem.strip()
        if recovered_title:
            return recovered_title

    return path.stem or path.name


def _historical_document_display_title(
    db: Session,
    document: Document | None,
) -> str | None:
    if document is None or not _is_storage_identity_title(str(document.title or "").strip()):
        return None
    jobs = list(
        db.scalars(
            select(IngestionJob)
            .where(IngestionJob.document_id == document.id)
            .order_by(IngestionJob.created_at.asc(), IngestionJob.id.asc())
            .limit(256)
        ).all()
    )
    for job in jobs:
        intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
        candidates = [
            ((intent.get("candidate_state") or {}).get("metadata") or {}).get("title"),
            ((intent.get("before_state") or {}).get("metadata") or {}).get("title"),
        ]
        for value in candidates:
            title = str(value or "").strip()
            if title and len(title) <= 255 and not _is_storage_identity_title(title):
                return title
    return None


def _document_partition(
    path: Path,
    *,
    knowledge_base_name: str,
    document: Document | None = None,
    display_filename: str | None = None,
    display_title: str | None = None,
    source_slot_protocol_version: str | None = None,
) -> str:
    effective_slot_protocol = source_slot_protocol_version or (
        document.source_slot_protocol_version if document is not None else None
    )
    if display_filename is not None or effective_slot_protocol == UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION:
        return _document_display_title(
            path,
            document=document,
            display_filename=display_filename,
            display_title=display_title,
            source_slot_protocol_version=effective_slot_protocol,
        )
    return derive_partition(path, knowledge_base_name=knowledge_base_name)


def _candidate_document_metadata(
    document: Document | None,
    knowledge_base: KnowledgeBase,
    path: Path,
    *,
    checksum: str,
    source_type: str,
    explicit_language: str | None = None,
    display_filename: str | None = None,
    display_title: str | None = None,
) -> dict[str, Any]:
    partition = _document_partition(
        path,
        knowledge_base_name=knowledge_base.name,
        document=document,
        display_filename=display_filename,
        display_title=display_title,
    )
    pending_language = pending_language_record_fields(explicit_language)
    return {
        "title": _document_display_title(
            path,
            document=document,
            display_filename=display_filename,
            display_title=display_title,
        ),
        "source_path": str(path),
        "source_type": source_type,
        **pending_language,
        "tags": [partition] if partition else [],
        "difficulty": document.difficulty if document is not None else None,
        "checksum": checksum,
        "visibility": document.visibility if document is not None else "private",
        "is_active": True,
    }


def _document_fact_scope(db: Session, knowledge_base_id: str, document_id: str | None) -> dict[str, Any]:
    if not document_id:
        return {
            "active_document_version_ids": [],
            "active_chunk_ids": [],
            "knowledge_base_chunk_version": int(
                db.scalar(select(KnowledgeBase.current_chunk_version).where(KnowledgeBase.id == knowledge_base_id)) or 0
            ),
        }
    return {
        "active_document_version_ids": sorted(
            str(item)
            for item in db.scalars(
                select(DocumentVersion.id).where(
                    DocumentVersion.document_id == document_id,
                    DocumentVersion.is_active.is_(True),
                )
            ).all()
        ),
        "active_chunk_ids": sorted(
            str(item)
            for item in db.scalars(
                select(Chunk.id).where(
                    Chunk.document_id == document_id,
                    Chunk.state == "active",
                )
            ).all()
        ),
        "knowledge_base_chunk_version": int(
            db.scalar(select(KnowledgeBase.current_chunk_version).where(KnowledgeBase.id == knowledge_base_id)) or 0
        ),
    }


def _build_document_metadata_intent(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    document: Document | None,
    path: Path,
    checksum: str,
    source_type: str,
    explicit_language: str | None = None,
    display_filename: str | None = None,
    display_title: str | None = None,
) -> dict[str, Any]:
    before_metadata = _document_metadata_payload(document) if document is not None else None
    before_state = {
        "exists": document is not None,
        "document_id": document.id if document is not None else None,
        "metadata": before_metadata,
        "created_at": document.created_at.isoformat() if document is not None and document.created_at else None,
        "updated_at": document.updated_at.isoformat() if document is not None and document.updated_at else None,
        **_document_fact_scope(db, knowledge_base.id, document.id if document is not None else None),
    }
    candidate_metadata = _candidate_document_metadata(
        document,
        knowledge_base,
        path,
        checksum=checksum,
        source_type=source_type,
        explicit_language=explicit_language,
        display_filename=display_filename,
        display_title=display_title,
    )
    candidate_state = {"metadata": candidate_metadata}
    return _refresh_pending_intent_payload_hash({
        "protocol_version": DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION,
        "schema_hash": document_metadata_intent_schema_hash(),
        "intent_id": str(uuid4()),
        "knowledge_base_id": knowledge_base.id,
        "source_path": str(path),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "document_id": document.id if document is not None else None,
        "before_state": before_state,
        "before_state_hash": _canonical_payload_hash(before_state),
        "candidate_state": candidate_state,
        "candidate_state_hash": _canonical_payload_hash(candidate_state),
        "candidate_revision": 1,
        "candidate_history": [],
        "pending_payload_hash": "",
    })


def _validate_document_metadata_intent(
    intent: dict[str, Any],
    *,
    knowledge_base_id: str,
    path: Path,
    checksum: str,
    source_type: str,
) -> None:
    if set(intent) != set(DOCUMENT_METADATA_PENDING_INTENT_FIELDS):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata intent envelope schema is invalid")
    if intent.get("protocol_version") != DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION:
        raise DocumentMetadataRestoreError("Queued ingestion job has an unsupported document metadata intent protocol")
    if intent.get("schema_hash") != document_metadata_intent_schema_hash():
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata intent schema hash is invalid")
    if intent.get("pending_payload_hash") != _pending_intent_payload_hash(intent):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata intent payload hash is invalid")
    if intent.get("status") != "pending":
        raise DocumentMetadataRestoreError("Queued ingestion job does not have a pending document metadata intent")
    if intent.get("knowledge_base_id") != knowledge_base_id or intent.get("source_path") != str(path):
        raise DocumentMetadataRestoreError("Queued ingestion job metadata intent does not match its knowledge base/source path")
    if not isinstance(intent.get("before_state"), dict):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata before-state must be an object")
    before_state = dict(intent["before_state"])
    if set(before_state) != set(DOCUMENT_METADATA_BEFORE_STATE_FIELDS):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata before-state schema is invalid")
    if intent.get("before_state_hash") != _canonical_payload_hash(before_state):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata before-state hash is invalid")
    if not isinstance(before_state.get("exists"), bool):
        raise DocumentMetadataRestoreError("Queued ingestion job before-state exists flag must be boolean")
    before_document_id = before_state.get("document_id")
    if before_state["exists"]:
        if not isinstance(before_document_id, str) or not before_document_id:
            raise DocumentMetadataRestoreError("Queued ingestion job existing before-state requires a document id")
        _validate_document_metadata_schema(before_state.get("metadata"), label="Document metadata before-state")
    elif before_document_id is not None or before_state.get("metadata") is not None:
        raise DocumentMetadataRestoreError("Queued ingestion job absent before-state cannot contain document metadata")
    for field in ("active_document_version_ids", "active_chunk_ids"):
        values = before_state.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
            raise DocumentMetadataRestoreError(f"Queued ingestion job before-state {field} must be a string list")
    if not isinstance(before_state.get("knowledge_base_chunk_version"), int) or before_state["knowledge_base_chunk_version"] < 0:
        raise DocumentMetadataRestoreError("Queued ingestion job before-state knowledge-base version is invalid")
    for field in ("created_at", "updated_at"):
        if before_state[field] is not None and not isinstance(before_state[field], str):
            raise DocumentMetadataRestoreError(f"Queued ingestion job before-state {field} must be a string or null")

    if not isinstance(intent.get("candidate_state"), dict):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata candidate-state must be an object")
    candidate_state = dict(intent["candidate_state"])
    if set(candidate_state) != set(DOCUMENT_METADATA_CANDIDATE_STATE_FIELDS):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata candidate-state schema is invalid")
    candidate = _validate_document_metadata_schema(
        candidate_state.get("metadata"),
        label="Document metadata candidate-state",
    )
    if candidate.get("source_path") != str(path):
        raise DocumentMetadataRestoreError("Queued ingestion job candidate source path does not match its intent")
    if candidate.get("checksum") != checksum or candidate.get("source_type") != source_type:
        raise DocumentMetadataRestoreError("Queued ingestion source changed after its document metadata intent was staged")
    if intent.get("candidate_state_hash") != _canonical_payload_hash(candidate_state):
        raise DocumentMetadataRestoreError("Queued ingestion job document metadata candidate hash is invalid")
    if not isinstance(intent.get("candidate_revision"), int) or int(intent["candidate_revision"]) < 1:
        raise DocumentMetadataRestoreError("Queued ingestion job candidate revision is invalid")
    if not isinstance(intent.get("candidate_history"), list):
        raise DocumentMetadataRestoreError("Queued ingestion job candidate history must be a list")
    for history_item in intent["candidate_history"]:
        if not isinstance(history_item, dict) or set(history_item) != set(DOCUMENT_METADATA_CANDIDATE_HISTORY_FIELDS):
            raise DocumentMetadataRestoreError("Queued ingestion job candidate history schema is invalid")
        if not isinstance(history_item["revision"], int) or history_item["revision"] < 1:
            raise DocumentMetadataRestoreError("Queued ingestion job candidate history revision is invalid")
        if not all(isinstance(history_item[field], str) and history_item[field] for field in ("candidate_state_hash", "replaced_at")):
            raise DocumentMetadataRestoreError("Queued ingestion job candidate history values are invalid")


def _apply_document_metadata(document: Document, metadata: dict[str, Any]) -> None:
    metadata = _validate_document_metadata_schema(metadata, label="Document metadata restore payload")
    document.title = str(metadata["title"])
    document.source_path = str(metadata["source_path"])
    document.source_type = str(metadata["source_type"])
    document.language = str(metadata["language"]) if metadata["language"] is not None else None
    document.language_source = (
        str(metadata["language_source"])
        if metadata["language_source"] is not None
        else None
    )
    document.language_confidence = (
        float(metadata["language_confidence"])
        if metadata["language_confidence"] is not None
        else None
    )
    document.language_detection_protocol_version = (
        str(metadata["language_detection_protocol_version"])
        if metadata["language_detection_protocol_version"] is not None
        else None
    )
    document.language_detection_hash = (
        str(metadata["language_detection_hash"])
        if metadata["language_detection_hash"] is not None
        else None
    )
    document.language_metadata_json = dict(metadata["language_metadata_json"] or {})
    document.tags = [str(item) for item in (metadata["tags"] or [])]
    document.difficulty = str(metadata["difficulty"]) if metadata["difficulty"] is not None else None
    document.checksum = str(metadata["checksum"])
    document.visibility = str(metadata["visibility"])
    document.is_active = bool(metadata["is_active"])


def _resolve_and_stage_document_language_identity(
    *,
    job: IngestionJob,
    document: Document,
    version: DocumentVersion,
    source_file: SourceFile,
    parse_job: ParseJob,
    sections: list[Any],
    knowledge_base_id: str,
    path: Path,
    checksum: str,
    source_type: str,
) -> dict[str, Any]:
    """Resolve one immutable language card inside the parse promotion transaction."""

    intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
    if intent.get("status") != "pending":
        raise DocumentMetadataRestoreError(
            "Language detection requires a pending document metadata intent"
        )
    candidate_state = dict(intent.get("candidate_state") or {})
    candidate_metadata = dict(candidate_state.get("metadata") or {})
    _validate_document_metadata_schema(
        candidate_metadata,
        label="Document language candidate metadata",
    )
    explicit_language = explicit_language_from_pending_metadata(
        candidate_metadata.get("language_metadata_json")
    )
    identity = detect_document_language(
        sections,
        explicit_language=explicit_language,
    )
    resolved_metadata = {**candidate_metadata, **identity}
    resolved_state = {"metadata": resolved_metadata}
    history = list(intent.get("candidate_history") or [])
    history.append(
        {
            "revision": int(intent.get("candidate_revision") or 1),
            "candidate_state_hash": _canonical_payload_hash(candidate_state),
            "replaced_at": datetime.utcnow().isoformat(),
        }
    )
    resolved_intent = _refresh_pending_intent_payload_hash(
        {
            **intent,
            "candidate_state": resolved_state,
            "candidate_state_hash": _canonical_payload_hash(resolved_state),
            "candidate_revision": int(intent.get("candidate_revision") or 1) + 1,
            "candidate_history": history[-16:],
        }
    )
    _validate_document_metadata_intent(
        resolved_intent,
        knowledge_base_id=knowledge_base_id,
        path=path,
        checksum=checksum,
        source_type=source_type,
    )
    _apply_document_metadata(document, resolved_metadata)
    apply_language_identity(version, identity)
    document_identity = language_identity_from_record(document)
    version_identity = language_identity_from_record(version)
    if (
        not document_identity["valid"]
        or not version_identity["valid"]
        or document_identity["detection_hash"]
        != version_identity["detection_hash"]
    ):
        raise DocumentMetadataRestoreError(
            "Document and active DocumentVersion language identities are not consistent"
        )
    source_file.metadata_json = {
        **(source_file.metadata_json or {}),
        "language_identity": dict(identity["language_metadata_json"]),
        "language_detection_hash": identity["language_detection_hash"],
    }
    summary = language_identity_summary(identity)
    parse_job.stats_json = {
        **(parse_job.stats_json or {}),
        "language_identity": summary,
    }
    parse_job.diagnostics_json = {
        **(parse_job.diagnostics_json or {}),
        "language_identity": dict(identity["language_metadata_json"]),
        "language_detection_hash": identity["language_detection_hash"],
    }
    job.stats = {
        **(job.stats or {}),
        DOCUMENT_METADATA_INTENT_KEY: resolved_intent,
        "language_identity": summary,
    }
    return identity


def current_worker_id() -> str:
    return os.getenv("WORKER_ID") or os.getenv("HOSTNAME") or socket.gethostname()


def exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message or f"{exc.__class__.__name__}: {exc!r}"


def graph_build_failure_audit(exc: BaseException) -> dict[str, Any]:
    """Return content-free graph/provider failure evidence for durable batch audit."""

    return {
        "protocol_version": "graph_build_failure_audit_v1",
        "failure_type": type(exc).__name__[:128],
        "external_failure": external_failure_classification(exc),
        "concept_provider_failure": concept_provider_output_failure_card(exc),
        "provider_response_persisted": False,
    }


def plan_versioned_graph_retry(
    db: Session,
    *,
    batch_id: str,
) -> dict[str, Any]:
    """Plan one exact graph-only retry for a committed version increment.

    It requires an exact pre-existing active graph before-image and a
    content-free durable failure audit, so a normal same-version maintenance
    rebuild cannot masquerade as the version-incrementing lineage.
    """

    batch = db.get(IngestionBatch, batch_id)
    if batch is None:
        raise RuntimeError("Versioned graph retry batch does not exist")
    recovery = db.scalar(
        select(IngestionBatchRecovery).where(
            IngestionBatchRecovery.batch_id == batch_id
        )
    )
    if recovery is None:
        raise RuntimeError("Versioned graph retry recovery row does not exist")
    _before_state, graph_before = _validate_batch_recovery(db, recovery)
    if (
        batch.status != "failed"
        or recovery.status != "graph_compensated"
        or not recovery.parse_committed
        or int(recovery.target_version) <= int(recovery.v_before_batch)
        or int(batch.total_files or 0) <= 0
        or int(batch.processed_files or 0) != int(batch.total_files or 0)
        or int(batch.success_count or 0) != int(batch.total_files or 0)
        or int(batch.failure_count or 0) != 0
        or int(batch.skipped_count or 0) != 0
    ):
        raise RuntimeError(
            "Versioned graph retry requires one failed, fully parsed, "
            "graph-compensated version increment"
        )
    graph_before_states = dict(graph_before.get("states") or {})
    if not any(
        graph_before_states.get(key)
        for key in ("relation", "mid", "coarse", "context")
    ):
        raise RuntimeError(
            "Versioned graph retry requires a pre-existing active graph before-image"
        )
    current_graph = _graph_before_state(db, recovery.knowledge_base_id)
    current_graph_hash = _canonical_payload_hash(current_graph)
    if current_graph_hash != recovery.graph_before_state_hash:
        raise RuntimeError(
            "Versioned graph retry active graph/pointer drifted from its exact before-image"
        )
    knowledge_base = db.get(KnowledgeBase, recovery.knowledge_base_id)
    if (
        knowledge_base is None
        or int(knowledge_base.current_chunk_version or 0)
        != int(recovery.target_version)
    ):
        raise RuntimeError(
            "Versioned graph retry knowledge-base version drifted from its retained target"
        )
    chunks = list(
        db.scalars(
            active_chunks_query(recovery.knowledge_base_id).order_by(
                Chunk.document_id.asc(), Chunk.chunk_index.asc(), Chunk.id.asc()
            )
        ).all()
    )
    if not chunks or any(
        int(chunk.chunk_version) != int(recovery.target_version)
        for chunk in chunks
    ):
        raise RuntimeError(
            "Versioned graph retry active chunks are not the exact retained target version"
        )
    stages = list(
        db.scalars(
            select(IngestionFileStage)
            .where(IngestionFileStage.batch_recovery_id == recovery.id)
            .order_by(
                IngestionFileStage.sequence_index.asc(),
                IngestionFileStage.id.asc(),
            )
        ).all()
    )
    if len(stages) != int(batch.total_files):
        raise RuntimeError(
            "Versioned graph retry file-stage cardinality drifted from the batch"
        )
    staged_chunk_ids: list[str] = []
    file_stage_write_set_hashes: list[str] = []
    for stage in stages:
        if stage.status != "retained_after_parse_commit":
            raise RuntimeError(
                "Versioned graph retry found a file stage outside the retained parse scope"
            )
        _stage_before, write_set = _validated_file_stage(
            db,
            stage,
            recovery=recovery,
            require_write_set=True,
        )
        staged_chunk_ids.extend(
            str(chunk_id) for chunk_id in (write_set.get("chunk_ids") or [])
        )
        file_stage_write_set_hashes.append(str(stage.write_set_hash))
    active_chunk_ids = sorted(str(chunk.id) for chunk in chunks)
    if sorted(staged_chunk_ids) != active_chunk_ids:
        raise RuntimeError(
            "Versioned graph retry retained file write sets do not equal the active chunk scope"
        )
    chunk_scope_hash = compute_chunk_scope_hash(chunks)
    failure_audit = dict((batch.stats or {}).get("graph_failure_audit") or {})
    if (
        failure_audit.get("protocol_version") != "graph_build_failure_audit_v1"
        or failure_audit.get("provider_response_persisted") is not False
    ):
        raise RuntimeError(
            "Versioned graph retry requires a content-free durable graph failure audit"
        )
    payload = {
        "protocol_version": VERSIONED_GRAPH_RETRY_PROTOCOL_VERSION,
        "operation": "same_batch_graph_only_retry",
        "batch_id": str(batch.id),
        "recovery_id": str(recovery.id),
        "knowledge_base_id": str(recovery.knowledge_base_id),
        "v_before_batch": int(recovery.v_before_batch),
        "target_version": int(recovery.target_version),
        "active_chunk_count": len(chunks),
        "active_chunk_ids_hash": _canonical_payload_hash(active_chunk_ids),
        "file_stage_write_set_hashes": file_stage_write_set_hashes,
        "chunk_scope_hash": chunk_scope_hash,
        "graph_before_state_hash": str(recovery.graph_before_state_hash),
        "current_graph_before_state_hash": current_graph_hash,
        "graph_failure_audit_hash": _canonical_payload_hash(failure_audit),
        "contextual_index_repair_mode": (
            CONTEXTUAL_INDEX_REPAIR_MODE_VERIFY_ONLY
        ),
        "provider_response_persisted": False,
        "gray_zone_model_call_count": 0,
    }
    return {**payload, "plan_hash": _canonical_payload_hash(payload)}


def is_active_file_job(job: IngestionJob) -> bool:
    if job.status == "queued" and not job.batch_id:
        return False
    return job.status in ACTIVE_FILE_JOB_STATES


def should_include_file(path: Path, *, authorized_root: Path) -> bool:
    lexical_path = Path(os.path.abspath(path))
    lexical_root = Path(os.path.abspath(authorized_root))
    try:
        relative_path = lexical_path.relative_to(lexical_root)
    except ValueError:
        return False
    if not path.is_file():
        return False
    if path.name.lower() in IGNORED_NAMES or path.name.startswith("~$"):
        return False
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False
    return not any(part.lower() in EXCLUDED_PARTS for part in relative_path.parts)


def collect_source_documents(
    root: Path,
    *,
    raw_manifest: bool = False,
) -> list[Path]:
    if raw_manifest:
        return load_raw_source_manifest(root)["paths"]
    if not root.exists():
        return []
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if should_include_file(path, authorized_root=root)
        ),
        key=lambda item: str(item).lower(),
    )
    deduped: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        validated = validate_source_content_path(
            path,
            root,
            max_bytes=get_settings().upload_max_bytes,
        )
        key = (path.name.lower(), validated.checksum)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def load_raw_source_manifest(
    import_root: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one manifest and all files under one frozen root identity."""

    lexical_import_root = Path(os.path.abspath(import_root))
    if lexical_import_root.is_symlink() or not lexical_import_root.is_dir():
        raise UploadValidationError("Raw import root must be one regular directory")
    try:
        with freeze_verified_readonly_import_root(lexical_import_root):
            return _load_raw_source_manifest_under_frozen_root(
                lexical_import_root,
                manifest_path=manifest_path,
            )
    except UploadValidationError:
        raise
    except SourceSnapshotError as exc:
        if "root identity changed" in str(exc).casefold():
            raise UploadValidationError(
                "Raw import root identity changed during manifest verification"
            ) from exc
        raise UploadValidationError(
            "Raw import root failed read-only final-open verification"
        ) from exc


def _load_raw_source_manifest_under_frozen_root(
    import_root: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve only checksum-bound raw files explicitly listed by a manifest."""

    lexical_import_root = Path(os.path.abspath(import_root))
    selected_manifest = Path(
        os.path.abspath(
            manifest_path or (lexical_import_root / RAW_SOURCE_MANIFEST_FILENAME)
        )
    )
    try:
        selected_manifest.relative_to(lexical_import_root)
    except ValueError as exc:
        raise UploadValidationError("Raw import manifest is outside its import root") from exc
    if selected_manifest.is_symlink() or not selected_manifest.is_file():
        raise UploadValidationError(
            f"Raw import requires {RAW_SOURCE_MANIFEST_FILENAME}"
        )
    try:
        with open_verified_readonly_import_file(
            selected_manifest,
            lexical_import_root,
        ) as (manifest_handle, _manifest_identity):
            manifest_bytes = manifest_handle.read(
                RAW_SOURCE_MANIFEST_MAX_BYTES + 1
            )
        if len(manifest_bytes) > RAW_SOURCE_MANIFEST_MAX_BYTES:
            raise UploadValidationError(
                "Raw import manifest exceeds its hard byte bound"
            )
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except UploadValidationError:
        raise
    except SourceSnapshotError as exc:
        if "root identity changed" in str(exc).casefold():
            raise UploadValidationError(
                "Raw import root identity changed during manifest verification"
            ) from exc
        raise UploadValidationError(
            "Raw import manifest failed final-open verification"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UploadValidationError("Raw import manifest is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "protocol_version",
        "raw_root",
        "files",
    }:
        raise UploadValidationError("Raw import manifest has an invalid frozen schema")
    if payload["protocol_version"] != RAW_SOURCE_MANIFEST_PROTOCOL_VERSION:
        raise UploadValidationError("Raw import manifest protocol is unsupported")
    raw_root_value = payload["raw_root"]
    if (
        not isinstance(raw_root_value, str)
        or not raw_root_value
        or Path(raw_root_value).is_absolute()
        or ".." in Path(raw_root_value).parts
    ):
        raise UploadValidationError("Raw import manifest raw_root must be relative")
    raw_root = Path(
        os.path.abspath(lexical_import_root / raw_root_value)
    )
    try:
        raw_root.relative_to(lexical_import_root)
    except ValueError as exc:
        raise UploadValidationError("Raw import root escapes the mounted import root") from exc
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise UploadValidationError("Raw import manifest raw_root is unavailable")
    files = payload["files"]
    if not isinstance(files, list) or len(files) > RAW_SOURCE_MANIFEST_MAX_FILES:
        raise UploadValidationError("Raw import manifest file count exceeds its hard bound")

    selected: list[Path] = []
    seen_keys: set[str] = set()
    seen_upload_names: set[str] = set()
    manifest_cards: list[dict[str, str]] = []
    for index, card in enumerate(files):
        if not isinstance(card, dict) or set(card) != {"path", "sha256"}:
            raise UploadValidationError(
                f"Raw import manifest file {index} has an invalid schema"
            )
        relative_value = card["path"]
        expected_checksum = card["sha256"]
        if (
            not isinstance(relative_value, str)
            or not relative_value
            or Path(relative_value).is_absolute()
            or ".." in Path(relative_value).parts
        ):
            raise UploadValidationError(
                f"Raw import manifest file {index} path is invalid"
            )
        if (
            not isinstance(expected_checksum, str)
            or len(expected_checksum) != 64
            or any(character not in "0123456789abcdef" for character in expected_checksum)
        ):
            raise UploadValidationError(
                f"Raw import manifest file {index} checksum is invalid"
            )
        normalized_key = unicodedata.normalize(
            "NFKC",
            Path(relative_value).as_posix(),
        ).casefold()
        if normalized_key in seen_keys:
            raise UploadValidationError(
                "Raw import manifest contains a casefold/NFKC path collision"
            )
        seen_keys.add(normalized_key)
        normalized_parts = [
            unicodedata.normalize("NFKC", part).casefold()
            for part in Path(relative_value).parts
        ]
        if (
            any(part in RAW_IMPORT_FORBIDDEN_PARTS for part in normalized_parts)
            or any(
                marker in normalized_parts[-1]
                for marker in (".uploading", ".previous", ".snapshotting")
            )
        ):
            raise UploadValidationError(
                f"Raw import manifest file {index} enters a derived/reserved path"
            )
        upload_name_key = unicodedata.normalize(
            "NFKC",
            Path(relative_value).name,
        ).casefold()
        if upload_name_key in seen_upload_names:
            raise UploadValidationError(
                "Raw import manifest contains a logical upload-slot filename collision"
            )
        seen_upload_names.add(upload_name_key)
        candidate = Path(os.path.abspath(raw_root / relative_value))
        try:
            candidate.relative_to(raw_root)
        except ValueError as exc:
            raise UploadValidationError(
                f"Raw import manifest file {index} escapes raw_root"
            ) from exc
        cursor = raw_root
        for component in Path(relative_value).parts:
            cursor /= component
            if cursor.is_symlink():
                raise UploadValidationError(
                    f"Raw import manifest file {index} traverses a symbolic link"
                )
        if not candidate.is_file() or not should_include_file(
            candidate,
            authorized_root=raw_root,
        ):
            raise UploadValidationError(
                f"Raw import manifest file {index} is not an allowed raw file"
            )
        try:
            observed_checksum, _source_identity = verified_readonly_import_checksum(
                candidate,
                lexical_import_root,
            )
        except SourceSnapshotError as exc:
            if "root identity changed" in str(exc).casefold():
                raise UploadValidationError(
                    "Raw import root identity changed during manifest verification"
                ) from exc
            raise UploadValidationError(
                f"Raw import manifest file {index} failed final-open verification"
            ) from exc
        if observed_checksum != expected_checksum:
            raise UploadValidationError(
                f"Raw import manifest checksum mismatch at file {index}"
            )
        selected.append(candidate)
        manifest_cards.append(
            {
                "path": Path(relative_value).as_posix(),
                "sha256": observed_checksum,
            }
        )
    return {
        "protocol_version": RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
        "manifest_path": str(selected_manifest),
        "raw_root": str(raw_root),
        "paths": selected,
        "files": manifest_cards,
        "file_count": len(selected),
        "manifest_hash": _canonical_payload_hash(
            {
                "protocol_version": RAW_SOURCE_MANIFEST_PROTOCOL_VERSION,
                "raw_root": Path(raw_root_value).as_posix(),
                "files": manifest_cards,
            }
        ),
    }


def resolve_knowledge_base(
    db: Session,
    knowledge_base_id: str | None = None,
    *,
    allow_deleting: bool = False,
) -> KnowledgeBase:
    if knowledge_base_id:
        knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    else:
        settings = get_settings()
        knowledge_base = db.scalar(select(KnowledgeBase).where(KnowledgeBase.name == settings.knowledge_base_name))
        if knowledge_base is None:
            knowledge_base = db.scalar(select(KnowledgeBase).order_by(KnowledgeBase.created_at.asc()))
    if knowledge_base is None:
        raise LookupError("Knowledge base not found")
    if (
        not allow_deleting
        and knowledge_base.lifecycle_status != "active"
    ):
        raise LookupError(
            "Knowledge base is unavailable while durable deletion recovery is pending"
        )
    return knowledge_base


def create_knowledge_base_space(db: Session, name: str, description: str | None = None) -> KnowledgeBase:
    existing = db.scalar(select(KnowledgeBase).where(KnowledgeBase.name == name))
    if existing is not None:
        raise ValueError(f"Knowledge base already exists: {name}")
    settings = get_settings()
    ensure_storage_durability_ready(settings=settings)
    knowledge_base_id = str(uuid4())
    paths = settings.knowledge_base_paths_for_id(knowledge_base_id)
    durable_ensure_directory(paths["storage_root"])
    durable_ensure_directory(paths["ingestion_root"])
    ensure_knowledge_base_storage_durability_ready(
        name,
        knowledge_base_id=knowledge_base_id,
    )
    knowledge_base = KnowledgeBase(
        id=knowledge_base_id,
        name=name,
        description=description,
        source_root=str(paths["storage_root"]),
    )
    db.add(knowledge_base)
    db.commit()
    db.refresh(knowledge_base)
    return knowledge_base


def list_knowledge_base_summaries(db: Session) -> list[dict]:
    return [
        summarize_knowledge_base(db, item)
        for item in db.scalars(
            select(KnowledgeBase)
            .where(KnowledgeBase.lifecycle_status == "active")
            .order_by(KnowledgeBase.created_at.asc())
        ).all()
    ]


def summarize_knowledge_base(db: Session, knowledge_base: KnowledgeBase) -> dict:
    paths = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )
    document_count = db.scalar(select(func.count(Document.id)).where(Document.knowledge_base_id == knowledge_base.id, Document.is_active.is_(True))) or 0
    graph_stats = context_graph_overview_stats(db, knowledge_base.id)
    graph_counts = graph_stats.get("counts") or {}
    freshness = graph_stats.get("freshness") or {}
    hashes = graph_stats.get("hashes") or {}
    chunk_count = int(graph_counts.get("chunks") or 0)
    active_chunk_count = int(graph_counts.get("active_chunks") or 0)
    stale_reason = ", ".join(freshness.get("stale_reasons") or []) or None
    return {
        "id": knowledge_base.id,
        "name": knowledge_base.name,
        "description": knowledge_base.description,
        "source_root": str(paths["storage_root"]),
        "storage_root": str(paths["storage_root"]),
        "document_count": document_count,
        "chunk_count": chunk_count,
        "active_chunk_count": active_chunk_count,
        "current_chunk_version": knowledge_base.current_chunk_version or 0,
        "context_graph_state_id": graph_stats.get("context_graph_state_id"),
        "context_graph_hash": hashes.get("context_graph_hash"),
        "stale_reason": stale_reason,
        "has_parsed_chunks": active_chunk_count > 0,
        "can_full_reparse": active_chunk_count > 0,
        "degraded_mode": is_degraded_mode(),
        "active_profile_id": knowledge_base.active_profile_id,
        "active_profile_name": knowledge_base.active_profile.name if knowledge_base.active_profile else None,
        "active_profile_hash": knowledge_base.active_profile.profile_hash if knowledge_base.active_profile else None,
    }


def create_job(
    db: Session,
    *,
    knowledge_base_id: str,
    document_id: str | None,
    trigger_source: str,
    batch_id: str | None = None,
    source_path: str | None = None,
    logical_source_slot_key: str | None = None,
    source_slot_protocol_version: str | None = None,
    stats: dict[str, Any] | None = None,
    commit: bool = True,
) -> IngestionJob:
    job = IngestionJob(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        trigger_source=trigger_source,
        batch_id=batch_id,
        source_path=source_path,
        logical_source_slot_key=logical_source_slot_key,
        source_slot_protocol_version=source_slot_protocol_version,
        status="queued",
        stats=dict(stats or {}),
    )
    db.add(job)
    if commit:
        db.commit()
        db.refresh(job)
    else:
        db.flush()
    return job


def resolve_source_slot_identity(
    db: Session,
    knowledge_base: KnowledgeBase,
    path: Path,
    *,
    logical_source_slot_key: str | None = None,
    source_slot_protocol_version: str | None = None,
) -> tuple[str, str, Document | None]:
    """Resolve one KB-scoped source identity and reject split ownership."""

    canonical_path = str(Path(path).resolve())
    path_documents = list(
        db.scalars(
            select(Document)
            .where(
                Document.knowledge_base_id == knowledge_base.id,
                Document.source_path == canonical_path,
            )
            .order_by(Document.id.asc())
            .limit(2)
        ).all()
    )
    if len(path_documents) > 1:
        raise DocumentMetadataRestoreError(
            "Multiple documents own the same physical source path"
        )

    if logical_source_slot_key is None:
        if path_documents:
            document = path_documents[0]
            return (
                document.logical_source_slot_key,
                document.source_slot_protocol_version,
                document,
            )
        storage_root = get_settings().knowledge_base_paths_for_source_root(
            knowledge_base.source_root
        )["storage_root"]
        logical_source_slot_key = source_slot_key_for_path(path, storage_root)
        source_slot_protocol_version = LOGICAL_SOURCE_SLOT_PROTOCOL_VERSION
    else:
        if (
            not isinstance(logical_source_slot_key, str)
            or not logical_source_slot_key
            or len(logical_source_slot_key) > 1024
        ):
            raise DocumentMetadataRestoreError("Logical source slot key is invalid")
        if not source_slot_protocol_version:
            raise DocumentMetadataRestoreError(
                "Logical source slot protocol version is required"
            )

    slot_documents = list(
        db.scalars(
            select(Document)
            .where(
                Document.knowledge_base_id == knowledge_base.id,
                Document.logical_source_slot_key == logical_source_slot_key,
            )
            .order_by(Document.id.asc())
            .limit(2)
        ).all()
    )
    if len(slot_documents) > 1:
        raise DocumentMetadataRestoreError(
            "Multiple documents own the same logical source slot"
        )
    path_document = path_documents[0] if path_documents else None
    slot_document = slot_documents[0] if slot_documents else None
    if (
        path_document is not None
        and slot_document is not None
        and path_document.id != slot_document.id
    ):
        raise DocumentMetadataRestoreError(
            "Physical source path and logical source slot have different owners"
        )
    document = slot_document or path_document
    if document is not None and (
        document.logical_source_slot_key != logical_source_slot_key
        or document.source_slot_protocol_version != source_slot_protocol_version
    ):
        raise DocumentMetadataRestoreError(
            "Existing source ownership does not match the requested logical slot"
        )
    return logical_source_slot_key, source_slot_protocol_version, document


def register_uploaded_file(
    db: Session,
    knowledge_base: KnowledgeBase,
    stored_path: Path,
    *,
    explicit_language: str | None = None,
    logical_source_slot_key: str | None = None,
    display_filename: str | None = None,
    commit: bool = True,
) -> tuple[Document, IngestionJob]:
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"]
    path = validate_knowledge_base_source_path(
        stored_path,
        knowledge_base_source_root=knowledge_base.source_root,
    )
    explicit_language = normalize_explicit_language_tag(explicit_language)
    if display_filename is not None:
        display_filename = normalize_upload_filename(display_filename)
        expected_upload_slot = normalize_upload_source_slot_key(display_filename)
        if logical_source_slot_key not in {None, expected_upload_slot}:
            raise DocumentMetadataRestoreError(
                "Upload filename does not match its logical source slot"
            )
        logical_source_slot_key = expected_upload_slot
        source_slot_protocol_version = UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION
    else:
        source_slot_protocol_version = (
            UPLOAD_SOURCE_SLOT_PROTOCOL_VERSION
            if logical_source_slot_key is not None
            else None
        )
    require_storage_durability_capability(path.parent)
    content_validation = validate_source_content_path(
        path,
        storage_root,
        max_bytes=get_settings().upload_max_bytes,
    )
    checksum = content_validation.checksum
    source_type = source_type_from_path(path)
    (
        logical_source_slot_key,
        source_slot_protocol_version,
        identity_document,
    ) = resolve_source_slot_identity(
        db,
        knowledge_base,
        path,
        logical_source_slot_key=logical_source_slot_key,
        source_slot_protocol_version=source_slot_protocol_version,
    )
    historical_display_title = (
        _historical_document_display_title(db, identity_document)
        if display_filename is None
        else None
    )
    reusable_job = _coalesce_pending_metadata_jobs(
        db,
        knowledge_base_id=knowledge_base.id,
        path=path,
        preferred_job=None,
    )
    if reusable_job is not None:
        intent = dict((reusable_job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
        existing_document = db.get(Document, intent.get("document_id"))
        if existing_document is None:
            raise DocumentMetadataRestoreError("Pending upload metadata candidate row is missing")
        previous_candidate_state = dict(intent["candidate_state"])
        candidate_metadata = _candidate_document_metadata(
            existing_document,
            knowledge_base,
            path,
            checksum=checksum,
            source_type=source_type,
            explicit_language=explicit_language,
            display_filename=display_filename,
            display_title=historical_display_title,
        )
        candidate_state = {"metadata": candidate_metadata}
        candidate_history = list(intent.get("candidate_history") or [])
        candidate_history.append(
            {
                "revision": int(intent.get("candidate_revision") or 1),
                "candidate_state_hash": _canonical_payload_hash(previous_candidate_state),
                "replaced_at": datetime.utcnow().isoformat(),
            }
        )
        revised_intent = _refresh_pending_intent_payload_hash({
            **intent,
            "document_id": existing_document.id,
            "candidate_state": candidate_state,
            "candidate_state_hash": _canonical_payload_hash(candidate_state),
            "candidate_revision": int(intent.get("candidate_revision") or 1) + 1,
            "candidate_history": candidate_history[-16:],
        })
        reusable_job.document_id = existing_document.id
        reusable_job.trigger_source = "upload"
        reusable_job.status = "queued"
        reusable_job.error_message = None
        reusable_job.stats = {**(reusable_job.stats or {}), DOCUMENT_METADATA_INTENT_KEY: revised_intent}
        if commit:
            db.commit()
            db.refresh(reusable_job)
        else:
            db.flush()
        return existing_document, reusable_job

    existing_document = identity_document
    intent = _build_document_metadata_intent(
        db,
        knowledge_base=knowledge_base,
        document=existing_document,
        path=path,
        checksum=checksum,
        source_type=source_type,
        explicit_language=explicit_language,
        display_filename=display_filename,
        display_title=historical_display_title,
    )
    document = existing_document or upsert_document(
        db,
        knowledge_base,
        path,
        checksum=checksum,
        source_type=source_type,
        logical_source_slot_key=logical_source_slot_key,
        source_slot_protocol_version=source_slot_protocol_version,
        display_filename=display_filename,
        display_title=historical_display_title,
    )
    intent["document_id"] = document.id
    intent = _refresh_pending_intent_payload_hash(intent)
    job = create_job(
        db,
        knowledge_base_id=knowledge_base.id,
        document_id=document.id,
        trigger_source="upload",
        source_path=str(path),
        logical_source_slot_key=logical_source_slot_key,
        source_slot_protocol_version=source_slot_protocol_version,
        stats={DOCUMENT_METADATA_INTENT_KEY: intent},
        commit=commit,
    )
    return document, job


def ingestion_job_language_identity_summary(job: IngestionJob) -> dict[str, Any]:
    intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
    candidate = dict((intent.get("candidate_state") or {}).get("metadata") or {})
    if candidate:
        return language_identity_summary(candidate)
    return {
        "status": "pending",
        "language": None,
        "source": None,
        "confidence": None,
        "protocol_version": None,
        "detection_hash": None,
        "explicit_language_tag": None,
        "decision_reason": None,
    }


def upsert_document(
    db: Session,
    knowledge_base: KnowledgeBase,
    path: Path,
    *,
    checksum: str,
    source_type: str,
    logical_source_slot_key: str | None = None,
    source_slot_protocol_version: str | None = None,
    display_filename: str | None = None,
    display_title: str | None = None,
) -> Document:
    (
        logical_source_slot_key,
        source_slot_protocol_version,
        document,
    ) = resolve_source_slot_identity(
        db,
        knowledge_base,
        path,
        logical_source_slot_key=logical_source_slot_key,
        source_slot_protocol_version=source_slot_protocol_version,
    )
    partition = _document_partition(
        path,
        knowledge_base_name=knowledge_base.name,
        document=document,
        display_filename=display_filename,
        display_title=display_title,
        source_slot_protocol_version=source_slot_protocol_version,
    )
    if document is None:
        document = Document(
            knowledge_base_id=knowledge_base.id,
            title=_document_display_title(
                path,
                display_filename=display_filename,
                display_title=display_title,
                logical_source_slot_key=logical_source_slot_key,
                source_slot_protocol_version=source_slot_protocol_version,
            ),
            source_path=str(path),
            logical_source_slot_key=logical_source_slot_key,
            source_slot_protocol_version=source_slot_protocol_version,
            source_type=source_type,
            tags=[partition] if partition else [],
            checksum=checksum,
            is_active=True,
        )
        db.add(document)
        db.flush()
    else:
        document.title = _document_display_title(
            path,
            document=document,
            display_filename=display_filename,
            display_title=display_title,
            logical_source_slot_key=logical_source_slot_key,
            source_slot_protocol_version=source_slot_protocol_version,
        )
        document.source_path = str(path)
        document.source_type = source_type
        document.checksum = checksum
        document.tags = [partition] if partition else []
        document.is_active = True
    return document


def _reusable_staged_ingestion_job(
    db: Session,
    *,
    knowledge_base_id: str,
    path: Path,
) -> IngestionJob | None:
    return _coalesce_pending_metadata_jobs(
        db,
        knowledge_base_id=knowledge_base_id,
        path=path,
        preferred_job=None,
    )


def _pending_metadata_candidate_checksum_for_snapshot(
    db: Session,
    *,
    knowledge_base_id: str,
    path: Path,
    existing_job_id: str | None,
) -> str | None:
    """Read the queued intent checksum before creating an immutable snapshot.

    The upload slot is mutable. When registration already persisted a pending
    candidate, the attempt may snapshot only those registered bytes. This
    lookup mirrors the pending-job choice made by metadata staging, but does not
    coalesce jobs, rewrite intents, or flush candidate metadata.
    """

    with db.no_autoflush:
        preferred_job = db.get(IngestionJob, existing_job_id) if existing_job_id else None
        if existing_job_id and preferred_job is None:
            raise LookupError(f"Ingestion job not found: {existing_job_id}")
        if preferred_job is not None and preferred_job.knowledge_base_id != knowledge_base_id:
            raise DocumentMetadataRestoreError("Ingestion job belongs to another knowledge base")

        queued_jobs = list(
            db.scalars(
                select(IngestionJob)
                .where(
                    IngestionJob.knowledge_base_id == knowledge_base_id,
                    IngestionJob.source_path == str(path),
                    IngestionJob.status == "queued",
                )
                .order_by(IngestionJob.created_at.asc(), IngestionJob.id.asc())
            ).all()
        )
        pending: list[tuple[IngestionJob, dict[str, Any]]] = []
        for job in queued_jobs:
            intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
            if (
                intent.get("protocol_version") != DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION
                or intent.get("status") != "pending"
            ):
                continue
            candidate = dict((intent.get("candidate_state") or {}).get("metadata") or {})
            _validate_document_metadata_intent(
                intent,
                knowledge_base_id=knowledge_base_id,
                path=path,
                checksum=str(candidate.get("checksum") or ""),
                source_type=str(candidate.get("source_type") or ""),
            )
            pending.append((job, intent))

        if preferred_job is not None:
            preferred_intent = dict(
                (preferred_job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {}
            )
            if preferred_intent.get("status") != "pending":
                return None
            chosen_pair = next(
                ((job, intent) for job, intent in pending if job.id == preferred_job.id),
                None,
            )
            if chosen_pair is None:
                raise DocumentMetadataRestoreError(
                    "Explicit ingestion job is not the active pending metadata intent for its source"
                )
        elif pending:
            chosen_pair = pending[-1]
        else:
            return None

        candidate = dict((chosen_pair[1].get("candidate_state") or {}).get("metadata") or {})
        checksum = str(candidate.get("checksum") or "")
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum.lower()):
            raise DocumentMetadataRestoreError(
                "Queued ingestion job candidate checksum is not a SHA-256 hex digest"
            )
        return checksum.lower()


def _coalesce_pending_metadata_jobs(
    db: Session,
    *,
    knowledge_base_id: str,
    path: Path,
    preferred_job: IngestionJob | None,
) -> IngestionJob | None:
    queued_jobs = list(db.scalars(
        select(IngestionJob)
        .where(
            IngestionJob.knowledge_base_id == knowledge_base_id,
            IngestionJob.source_path == str(path),
            IngestionJob.status == "queued",
        )
        .order_by(IngestionJob.created_at.asc(), IngestionJob.id.asc())
    ).all())
    pending: list[tuple[IngestionJob, dict[str, Any]]] = []
    unsupported_pending: list[dict[str, str | None]] = []
    for job in queued_jobs:
        intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
        if intent.get("status") != "pending":
            continue
        if intent.get("protocol_version") != DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION:
            unsupported_pending.append(
                {
                    "job_id": str(job.id),
                    "protocol_version": (
                        str(intent.get("protocol_version"))
                        if intent.get("protocol_version") is not None
                        else None
                    ),
                }
            )
            continue
        candidate = dict((intent.get("candidate_state") or {}).get("metadata") or {})
        _validate_document_metadata_intent(
            intent,
            knowledge_base_id=knowledge_base_id,
            path=path,
            checksum=str(candidate.get("checksum") or ""),
            source_type=str(candidate.get("source_type") or ""),
        )
        pending.append((job, intent))
    if unsupported_pending:
        raise DocumentMetadataRestoreError(
            "Queued ingestion contains a pending document metadata intent from an "
            "unsupported protocol; explicitly resolve or re-register it before "
            f"ingestion: {unsupported_pending}"
        )
    if not pending:
        return None

    canonical_before = dict(pending[0][1]["before_state"])
    canonical_before_hash = str(pending[0][1]["before_state_hash"])
    pending_jobs = {job.id: (job, intent) for job, intent in pending}
    if preferred_job is not None:
        chosen_pair = pending_jobs.get(preferred_job.id)
        if chosen_pair is None:
            raise DocumentMetadataRestoreError("Explicit ingestion job is not the active pending metadata intent for its source")
    else:
        chosen_pair = pending[-1]
    chosen_job, chosen_intent = chosen_pair
    chosen_intent = _refresh_pending_intent_payload_hash({
        **chosen_intent,
        "before_state": canonical_before,
        "before_state_hash": canonical_before_hash,
    })
    chosen_job.stats = {**(chosen_job.stats or {}), DOCUMENT_METADATA_INTENT_KEY: chosen_intent}
    for job, intent in pending:
        if job.id == chosen_job.id:
            continue
        job.status = "skipped"
        job.error_message = "superseded by a newer pending upload for the same source path"
        job.stats = {
            **(job.stats or {}),
            "phase": "metadata_intent_superseded",
            DOCUMENT_METADATA_INTENT_KEY: {
                **intent,
                "status": "superseded",
                "superseded_at": datetime.utcnow().isoformat(),
                "superseded_by_job_id": chosen_job.id,
            },
        }
    db.flush()
    return chosen_job


def _stage_document_metadata_for_ingestion(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    path: Path,
    checksum: str,
    source_type: str,
    trigger_source: str,
    batch_id: str | None,
    existing_job_id: str | None,
) -> tuple[Document, IngestionJob, dict[str, Any]]:
    (
        logical_source_slot_key,
        source_slot_protocol_version,
        identity_document,
    ) = resolve_source_slot_identity(db, knowledge_base, path)
    historical_display_title = _historical_document_display_title(
        db,
        identity_document,
    )
    job = db.get(IngestionJob, existing_job_id) if existing_job_id else None
    if existing_job_id and job is None:
        raise LookupError(f"Ingestion job not found: {existing_job_id}")
    if job is not None:
        current_intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
        if current_intent.get("status") == "pending":
            job = _coalesce_pending_metadata_jobs(
                db,
                knowledge_base_id=knowledge_base.id,
                path=path,
                preferred_job=job,
            )
    else:
        job = _reusable_staged_ingestion_job(
            db,
            knowledge_base_id=knowledge_base.id,
            path=path,
        )
    if job is not None and job.knowledge_base_id != knowledge_base.id:
        raise DocumentMetadataRestoreError("Ingestion job belongs to another knowledge base")

    pending_intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {}) if job is not None else {}
    if pending_intent.get("status") == "pending":
        _validate_document_metadata_intent(
            pending_intent,
            knowledge_base_id=knowledge_base.id,
            path=path,
            checksum=checksum,
            source_type=source_type,
        )
        document = db.get(Document, pending_intent.get("document_id"))
        if document is None:
            raise DocumentMetadataRestoreError("Pending document metadata candidate row is missing")
        before_state = dict(pending_intent.get("before_state") or {})
        if bool(before_state.get("exists")):
            applied_metadata_hash = _canonical_payload_hash(_document_metadata_payload(document))
            expected_metadata_hash = _canonical_payload_hash(before_state.get("metadata") or {})
            if applied_metadata_hash != expected_metadata_hash:
                raise DocumentMetadataRestoreError("Applied document metadata drifted before candidate promotion")
        candidate_metadata = dict((pending_intent.get("candidate_state") or {}).get("metadata") or {})
        job.batch_id = batch_id or job.batch_id
        job.document_id = document.id
        job.source_path = str(path)
        job.logical_source_slot_key = logical_source_slot_key
        job.source_slot_protocol_version = source_slot_protocol_version
        # The pending intent and its batch ownership must survive a hard worker
        # termination.  Commit only that control-plane association before
        # applying candidate metadata; candidate promotion remains in the main
        # DocumentVersion/chunk transaction below.
        db.commit()
        db.refresh(job)
        document = db.get(Document, pending_intent.get("document_id"))
        if document is None:
            raise DocumentMetadataRestoreError("Pending document metadata candidate row disappeared before promotion")
        _apply_document_metadata(document, candidate_metadata)
        return document, job, pending_intent

    existing_document = identity_document
    intent = _build_document_metadata_intent(
        db,
        knowledge_base=knowledge_base,
        document=existing_document,
        path=path,
        checksum=checksum,
        source_type=source_type,
        display_title=historical_display_title,
    )
    document = existing_document or upsert_document(
        db,
        knowledge_base,
        path,
        checksum=checksum,
        source_type=source_type,
        logical_source_slot_key=logical_source_slot_key,
        source_slot_protocol_version=source_slot_protocol_version,
        display_title=historical_display_title,
    )
    intent["document_id"] = document.id
    intent = _refresh_pending_intent_payload_hash(intent)
    if job is None:
        job = create_job(
            db,
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            trigger_source=trigger_source,
            batch_id=batch_id,
            source_path=str(path),
            logical_source_slot_key=logical_source_slot_key,
            source_slot_protocol_version=source_slot_protocol_version,
            stats={DOCUMENT_METADATA_INTENT_KEY: intent},
        )
    else:
        job.document_id = document.id
        job.trigger_source = trigger_source
        job.batch_id = batch_id or job.batch_id
        job.source_path = str(path)
        job.logical_source_slot_key = logical_source_slot_key
        job.source_slot_protocol_version = source_slot_protocol_version
        job.status = "queued"
        job.error_message = None
        job.stats = {**(job.stats or {}), DOCUMENT_METADATA_INTENT_KEY: intent}
        db.commit()
        db.refresh(job)
    candidate_metadata = dict((intent.get("candidate_state") or {}).get("metadata") or {})
    _apply_document_metadata(document, candidate_metadata)
    return document, job, intent


def _pending_metadata_intent_job(
    db: Session,
    *,
    knowledge_base_id: str,
    path: Path,
    existing_job_id: str | None,
) -> tuple[IngestionJob, dict[str, Any]] | None:
    candidates: list[IngestionJob] = []
    if existing_job_id:
        explicit = db.get(IngestionJob, existing_job_id)
        if explicit is not None:
            candidates.append(explicit)
    candidates.extend(
        db.scalars(
            select(IngestionJob)
            .where(
                IngestionJob.knowledge_base_id == knowledge_base_id,
                IngestionJob.source_path == str(path),
            )
            .order_by(IngestionJob.updated_at.desc())
        ).all()
    )
    seen: set[str] = set()
    for job in candidates:
        if job.id in seen:
            continue
        seen.add(job.id)
        intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
        if (
            intent.get("protocol_version") == DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION
            and intent.get("status") == "pending"
            and intent.get("knowledge_base_id") == knowledge_base_id
            and intent.get("source_path") == str(path)
        ):
            candidate = dict((intent.get("candidate_state") or {}).get("metadata") or {})
            _validate_document_metadata_intent(
                intent,
                knowledge_base_id=knowledge_base_id,
                path=path,
                checksum=str(candidate.get("checksum") or ""),
                source_type=str(candidate.get("source_type") or ""),
            )
            return job, intent
    return None


def _metadata_failure_message(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "asyncio ingestion task cancelled"
    if isinstance(exc, Exception):
        return exception_message(exc)[:2000]
    return exc.__class__.__name__


def _source_snapshot_recovery_identity(
    db: Session,
    *,
    job: IngestionJob,
    intent: dict[str, Any],
) -> dict[str, Any]:
    knowledge_base = db.get(KnowledgeBase, job.knowledge_base_id)
    if knowledge_base is None:
        raise DocumentMetadataRestoreError("Cannot resolve knowledge base for source snapshot audit")
    candidate = dict((intent.get("candidate_state") or {}).get("metadata") or {})
    checksum = str(candidate.get("checksum") or "").lower()
    source_path = str(intent.get("source_path") or job.source_path or "")
    if not source_path:
        raise DocumentMetadataRestoreError("Cannot derive source snapshot identity without a source path")
    snapshot_path = source_snapshot_path(
        Path(source_path).name,
        checksum,
        knowledge_base_source_root=knowledge_base.source_root,
        create_parents=False,
    )
    ingestion_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["ingestion_root"]
    try:
        frozen_snapshot = freeze_existing_source_snapshot(
            snapshot_path,
            authorized_root=ingestion_root,
            expected_checksum=checksum,
        )
    except SourceSnapshotNotFoundError:
        return {
            "protocol_version": SOURCE_SNAPSHOT_PROTOCOL_VERSION,
            "path": str(snapshot_path),
            "checksum": checksum,
            "exists": False,
            "checksum_verified": False,
            "frozen_snapshot_identity": None,
            "disposition": "retained_for_audit_or_retry",
        }
    return {
        "protocol_version": SOURCE_SNAPSHOT_PROTOCOL_VERSION,
        "path": str(snapshot_path),
        "checksum": checksum,
        "exists": True,
        "checksum_verified": True,
        "frozen_snapshot_identity": frozen_snapshot.identity_card(),
        "disposition": "retained_for_audit_or_retry",
    }


def _restore_pending_document_metadata_intent(
    db: Session,
    *,
    job: IngestionJob,
    intent: dict[str, Any],
    exc: BaseException,
) -> str:
    source_snapshot_identity: dict[str, Any] | None = None
    try:
        source_path = str(intent.get("source_path") or job.source_path or "")
        if not source_path:
            raise DocumentMetadataRestoreError("Cannot restore document metadata: source path is missing")
        path = Path(source_path)
        candidate = dict((intent.get("candidate_state") or {}).get("metadata") or {})
        _validate_document_metadata_intent(
            intent,
            knowledge_base_id=job.knowledge_base_id,
            path=path,
            checksum=str(candidate.get("checksum") or ""),
            source_type=str(candidate.get("source_type") or ""),
        )
        source_snapshot_identity = _source_snapshot_recovery_identity(
            db,
            job=job,
            intent=intent,
        )
        if (
            source_snapshot_identity["exists"]
            and not source_snapshot_identity["checksum_verified"]
        ):
            raise DocumentMetadataRestoreError(
                "Cannot restore document metadata while its immutable source snapshot fails checksum verification"
            )
        knowledge_base_id = job.knowledge_base_id
        before_state = dict(intent.get("before_state") or {})
        if intent.get("before_state_hash") != _canonical_payload_hash(before_state):
            raise DocumentMetadataRestoreError("Cannot restore document metadata: before-state hash verification failed")
        before_exists = bool(before_state.get("exists"))
        before_document_id = before_state.get("document_id")
        candidate_document_id = intent.get("document_id")
        absent_candidate_cleanup_guard: dict[str, Any] | None = None
        if before_exists:
            if not before_document_id:
                raise DocumentMetadataRestoreError("Cannot restore document metadata: before-state document id is missing")
            if candidate_document_id != before_document_id or job.document_id != before_document_id:
                raise DocumentMetadataRestoreError(
                    "Cannot restore document metadata: existing-document intent ownership is inconsistent"
                )
            document = db.get(Document, before_document_id)
            if document is None:
                raise DocumentMetadataRestoreError("Cannot restore document metadata: before-state document row is missing")
            before_metadata = dict(before_state.get("metadata") or {})
            _apply_document_metadata(document, before_metadata)
            if before_state.get("created_at"):
                document.created_at = datetime.fromisoformat(str(before_state["created_at"]))
            if before_state.get("updated_at"):
                document.updated_at = datetime.fromisoformat(str(before_state["updated_at"]))
            restored_document_id: str | None = document.id
            job.document_id = document.id
        else:
            candidate_document = db.get(Document, candidate_document_id) if candidate_document_id else None
            if candidate_document is None:
                raise DocumentMetadataRestoreError(
                    "Cannot restore document metadata: absent-before-state candidate row is missing"
                )
            if (
                job.document_id != candidate_document.id
                or candidate_document.knowledge_base_id != knowledge_base_id
                or candidate_document.source_path != str(path)
            ):
                raise DocumentMetadataRestoreError(
                    "Cannot restore document metadata: absent-before-state candidate ownership is inconsistent"
                )
            competing_job_ids = list(
                db.scalars(
                    select(IngestionJob.id).where(
                        IngestionJob.document_id == candidate_document.id,
                        IngestionJob.id != job.id,
                        IngestionJob.status.notin_(TERMINAL_STATES),
                    )
                ).all()
            )
            absent_candidate_cleanup_guard = {
                "document_version_count": int(
                    db.scalar(
                        select(func.count(DocumentVersion.id)).where(
                            DocumentVersion.document_id == candidate_document.id
                        )
                    )
                    or 0
                ),
                "chunk_count": int(
                    db.scalar(
                        select(func.count(Chunk.id)).where(
                            Chunk.document_id == candidate_document.id
                        )
                    )
                    or 0
                ),
                "source_file_count": int(
                    db.scalar(
                        select(func.count(SourceFile.id)).where(
                            SourceFile.document_id == candidate_document.id
                        )
                    )
                    or 0
                ),
                "competing_nonterminal_job_ids": sorted(str(item) for item in competing_job_ids),
            }
            absent_candidate_cleanup_guard["ok"] = not (
                absent_candidate_cleanup_guard["document_version_count"]
                or absent_candidate_cleanup_guard["chunk_count"]
                or absent_candidate_cleanup_guard["source_file_count"]
                or absent_candidate_cleanup_guard["competing_nonterminal_job_ids"]
            )
            if not absent_candidate_cleanup_guard["ok"]:
                raise DocumentMetadataRestoreError(
                    "Cannot remove an absent-before-state candidate document because it acquired "
                    f"committed or competing facts: {absent_candidate_cleanup_guard}"
                )
            db.delete(candidate_document)
            job.document_id = None
            restored_document_id = None
        db.flush()

        restored_scope = _document_fact_scope(db, knowledge_base_id, restored_document_id)
        expected_scope = {
            "active_document_version_ids": sorted(str(item) for item in (before_state.get("active_document_version_ids") or [])),
            "active_chunk_ids": sorted(str(item) for item in (before_state.get("active_chunk_ids") or [])),
            "knowledge_base_chunk_version": int(before_state.get("knowledge_base_chunk_version") or 0),
        }
        document_scope_matches = all(
            restored_scope[field] == expected_scope[field]
            for field in ("active_document_version_ids", "active_chunk_ids")
        )
        knowledge_base_version_not_regressed = (
            int(restored_scope["knowledge_base_chunk_version"])
            >= int(expected_scope["knowledge_base_chunk_version"])
        )
        verification = {
            "document_presence_matches": bool(
                (before_exists and restored_document_id and db.get(Document, restored_document_id) is not None)
                or (
                    not before_exists
                    and db.scalar(
                        select(Document.id).where(
                            Document.knowledge_base_id == knowledge_base_id,
                            Document.source_path == str(path),
                        )
                    )
                    is None
                )
            ),
            "metadata_hash_matches": bool(
                not before_exists
                or _canonical_payload_hash(_document_metadata_payload(db.get(Document, restored_document_id)))
                == _canonical_payload_hash(before_state.get("metadata") or {})
            ),
            "fact_scope_matches": document_scope_matches,
            "knowledge_base_version_not_regressed": knowledge_base_version_not_regressed,
            "knowledge_base_version_advanced": (
                int(restored_scope["knowledge_base_chunk_version"])
                > int(expected_scope["knowledge_base_chunk_version"])
            ),
            "restored_scope": restored_scope,
            "expected_scope": expected_scope,
            "source_snapshot_identity": source_snapshot_identity,
            "absent_candidate_cleanup_guard": absent_candidate_cleanup_guard,
        }
        verification["ok"] = all(
            bool(verification[key])
            for key in (
                "document_presence_matches",
                "metadata_hash_matches",
                "fact_scope_matches",
                "knowledge_base_version_not_regressed",
            )
        )
        if not verification["ok"]:
            raise DocumentMetadataRestoreError(
                f"Document metadata restoration verification failed: {verification}"
            )

        failure_message = _metadata_failure_message(exc)
        restored_intent = {
            **intent,
            "status": "restored",
            "restored_at": datetime.utcnow().isoformat(),
            "failure": {"type": exc.__class__.__name__, "message": failure_message},
            "restore_verification": verification,
            "source_snapshot": source_snapshot_identity,
        }
        job.status = "cancelled" if isinstance(exc, (IngestionCancelled, asyncio.CancelledError)) else "failed"
        job.error_message = failure_message
        job.stats = {
            **(job.stats or {}),
            "phase": "metadata_restored_after_failure",
            "source_snapshot_recovery": source_snapshot_identity,
            DOCUMENT_METADATA_INTENT_KEY: restored_intent,
        }
        db.commit()
        return job.id
    except Exception as restore_exc:
        db.rollback()
        failed_job = db.get(IngestionJob, job.id)
        if failed_job is not None:
            failed_intent = dict((failed_job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or intent)
            failed_job.status = "failed"
            failed_job.error_message = f"metadata restore failed: {restore_exc}"[:2000]
            failed_job.stats = {
                **(failed_job.stats or {}),
                "phase": "metadata_restore_failed",
                "manual_review_required": True,
                **(
                    {"source_snapshot_recovery": source_snapshot_identity}
                    if source_snapshot_identity is not None
                    else {}
                ),
                DOCUMENT_METADATA_INTENT_KEY: {
                    **failed_intent,
                    "status": "restore_failed",
                    "restore_failed_at": datetime.utcnow().isoformat(),
                    "restore_error": {
                        "type": restore_exc.__class__.__name__,
                        "message": str(restore_exc)[:2000],
                    },
                    **(
                        {"source_snapshot": source_snapshot_identity}
                        if source_snapshot_identity is not None
                        else {}
                    ),
                },
            }
            db.commit()
        raise DocumentMetadataRestoreError(
            f"Failed ingestion could not restore its document metadata before-state: {restore_exc}"
        ) from restore_exc


def _restore_document_metadata_after_failure(
    db: Session,
    *,
    knowledge_base_id: str,
    path: Path,
    existing_job_id: str | None,
    exc: BaseException,
) -> str | None:
    db.rollback()
    located = _pending_metadata_intent_job(
        db,
        knowledge_base_id=knowledge_base_id,
        path=path,
        existing_job_id=existing_job_id,
    )
    if located is None:
        return None
    job, intent = located
    return _restore_pending_document_metadata_intent(
        db,
        job=job,
        intent=intent,
        exc=exc,
    )


def _mark_document_metadata_intent_applied(
    db: Session,
    *,
    job: IngestionJob,
    document: Document,
) -> None:
    intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
    if intent.get("protocol_version") != DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION or intent.get("status") != "pending":
        raise DocumentMetadataRestoreError("Successful ingestion is missing a pending document metadata intent")
    actual_metadata = _document_metadata_payload(document)
    if intent.get("candidate_state_hash") != _canonical_payload_hash({"metadata": actual_metadata}):
        raise DocumentMetadataRestoreError("Successful ingestion document metadata does not match its staged candidate")
    applied_intent = {
        **intent,
        "status": "applied",
        "applied_at": datetime.utcnow().isoformat(),
        "apply_verification": {
            "ok": True,
            "metadata_hash": _canonical_payload_hash(actual_metadata),
            **_document_fact_scope(db, job.knowledge_base_id, document.id),
        },
    }
    job.stats = {**(job.stats or {}), DOCUMENT_METADATA_INTENT_KEY: applied_intent}


def _chunk_version_card(row: ChunkVersion) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "chunk_version": int(row.chunk_version),
        "chunk_schema_version": str(row.chunk_schema_version),
        "tokenizer_version": str(row.tokenizer_version),
        "chunk_size": int(row.chunk_size),
        "chunk_overlap": int(row.chunk_overlap),
        "state_hash": str(row.state_hash),
        "stats_json": dict(row.stats_json or {}),
        "diagnostics_json": dict(row.diagnostics_json or {}),
        "state": str(row.state),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _graph_state_hash(row: Any) -> str:
    value = getattr(row, "state_hash", None)
    if value is None and isinstance(row, ContextGraphState):
        value = row.context_graph_hash
    return str(value or "")


def _graph_before_state(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    state_cards: dict[str, list[dict[str, str]]] = {}
    for key, model in (
        ("relation", ChunkRelationGraphState),
        ("mid", MidConceptState),
        ("coarse", CoarseConceptState),
        ("context", ContextGraphState),
    ):
        rows = list(
            db.scalars(
                select(model)
                .where(
                    model.knowledge_base_id == knowledge_base_id,
                    model.state == "active",
                )
                .order_by(model.id.asc())
            ).all()
        )
        state_cards[key] = [
            {"id": str(row.id), "state_hash": _graph_state_hash(row)} for row in rows
        ]
    pointer = db.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id == knowledge_base_id
        )
    )
    pointer_card = None
    if pointer is not None:
        pointer_card = {
            "id": str(pointer.id),
            "state_hash": str(pointer.state_hash),
            "active_context_graph_state_id": pointer.active_context_graph_state_id,
            "active_chunk_relation_graph_state_id": pointer.active_chunk_relation_graph_state_id,
            "active_mid_concept_state_id": pointer.active_mid_concept_state_id,
            "active_coarse_concept_state_id": pointer.active_coarse_concept_state_id,
            "promotion_audit_json": dict(pointer.promotion_audit_json or {}),
        }
    return {
        "protocol_version": "ingestion_graph_before_scope_v1",
        "knowledge_base_id": str(knowledge_base_id),
        "states": state_cards,
        "vector_runtime_pointer": pointer_card,
    }


def _batch_before_state(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
) -> dict[str, Any]:
    active_chunks = list(
        db.scalars(
            active_chunks_query(knowledge_base.id)
            .order_by(Chunk.document_id.asc(), Chunk.chunk_index.asc(), Chunk.id.asc())
            .execution_options(populate_existing=True)
        ).all()
    )
    active_versions = list(
        db.scalars(
            select(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.knowledge_base_id == knowledge_base.id,
                DocumentVersion.is_active.is_(True),
            )
            .order_by(DocumentVersion.document_id.asc(), DocumentVersion.id.asc())
        ).all()
    )
    descriptors = list(
        db.scalars(
            select(ChunkVersion)
            .where(ChunkVersion.knowledge_base_id == knowledge_base.id)
            .order_by(ChunkVersion.chunk_version.asc(), ChunkVersion.id.asc())
        ).all()
    )
    return {
        "protocol_version": INGESTION_BATCH_RECOVERY_BEFORE_PROTOCOL_VERSION,
        "knowledge_base_id": str(knowledge_base.id),
        "v_before_batch": int(knowledge_base.current_chunk_version or 0),
        "active_chunk_ids": [str(chunk.id) for chunk in active_chunks],
        "active_chunk_scope_hash": compute_chunk_scope_hash(active_chunks),
        "active_document_version_ids": [str(row.id) for row in active_versions],
        "chunk_version_descriptors": [_chunk_version_card(row) for row in descriptors],
    }


def _validate_batch_recovery(
    db: Session,
    recovery: IngestionBatchRecovery,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if recovery.protocol_version != INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION:
        raise RuntimeError(f"Batch recovery {recovery.id} uses an unsupported protocol")
    expected_statuses = (
        INGESTION_RECOVERY_PARSE_COMMITTED_STATUSES
        if bool(recovery.parse_committed)
        else INGESTION_RECOVERY_PARSE_UNCOMMITTED_STATUSES
    )
    if recovery.status not in expected_statuses:
        raise RuntimeError(
            f"Batch recovery {recovery.id} status/parse-commit state is invalid"
        )
    if (
        int(recovery.v_before_batch) < 0
        or int(recovery.target_version) < int(recovery.v_before_batch)
    ):
        raise RuntimeError(f"Batch recovery {recovery.id} version boundary is invalid")
    with db.no_autoflush:
        batch = db.get(IngestionBatch, recovery.batch_id)
    if batch is None or batch.knowledge_base_id != recovery.knowledge_base_id:
        raise RuntimeError(
            f"Batch recovery {recovery.id} batch/knowledge-base ownership is invalid"
        )
    before_state = dict(recovery.before_state_json or {})
    graph_before = dict(recovery.graph_before_state_json or {})
    if (
        before_state.get("protocol_version")
        != INGESTION_BATCH_RECOVERY_BEFORE_PROTOCOL_VERSION
        or before_state.get("knowledge_base_id") != recovery.knowledge_base_id
        or int(before_state.get("v_before_batch") or 0) != int(recovery.v_before_batch)
        or _canonical_payload_hash(before_state) != recovery.before_state_hash
    ):
        raise RuntimeError(f"Batch recovery {recovery.id} before-state identity is invalid")
    if (
        graph_before.get("protocol_version") != "ingestion_graph_before_scope_v1"
        or graph_before.get("knowledge_base_id") != recovery.knowledge_base_id
        or _canonical_payload_hash(graph_before) != recovery.graph_before_state_hash
    ):
        raise RuntimeError(f"Batch recovery {recovery.id} graph before-state identity is invalid")
    return before_state, graph_before


def _prepare_batch_recovery(
    db: Session,
    *,
    batch: IngestionBatch,
    knowledge_base: KnowledgeBase,
    target_version: int,
    full_reparse: bool,
) -> IngestionBatchRecovery:
    existing = db.scalar(
        select(IngestionBatchRecovery)
        .where(IngestionBatchRecovery.batch_id == batch.id)
        .with_for_update()
    )
    if existing is not None:
        _validate_batch_recovery(db, existing)
        if (
            existing.knowledge_base_id != knowledge_base.id
            or int(existing.target_version) != int(target_version)
            or bool(existing.full_reparse) != bool(full_reparse)
        ):
            raise RuntimeError("Existing ingestion recovery row does not match this batch execution")
        return existing
    before_state = _batch_before_state(db, knowledge_base=knowledge_base)
    graph_before = _graph_before_state(db, knowledge_base.id)
    recovery = IngestionBatchRecovery(
        batch_id=batch.id,
        knowledge_base_id=knowledge_base.id,
        protocol_version=INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION,
        status="prepared",
        v_before_batch=int(knowledge_base.current_chunk_version or 0),
        target_version=int(target_version),
        full_reparse=bool(full_reparse),
        parse_committed=False,
        before_state_json=before_state,
        before_state_hash=_canonical_payload_hash(before_state),
        graph_before_state_json=graph_before,
        graph_before_state_hash=_canonical_payload_hash(graph_before),
        graph_write_set_json={},
        compensation_json={},
        diagnostics_json={
            "version_restore_source": "durable_v_before_batch",
            "version_minus_one_inference_used": False,
            "gray_zone_model_call_count": 0,
        },
    )
    db.add(recovery)
    db.flush()
    batch.stats = {
        **(batch.stats or {}),
        "batch_recovery_id": recovery.id,
        "batch_recovery_protocol_version": recovery.protocol_version,
        "v_before_batch": recovery.v_before_batch,
        "version_minus_one_inference_used": False,
    }
    db.commit()
    db.refresh(recovery)
    return recovery


def _source_file_card(row: SourceFile) -> dict[str, Any]:
    return {
        "id": str(row.id),
        "document_id": row.document_id,
        "source_path": str(row.source_path),
        "checksum": str(row.checksum),
        "source_type": str(row.source_type),
        "size_bytes": int(row.size_bytes or 0),
        "metadata_json": dict(row.metadata_json or {}),
        "state": str(row.state),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _file_before_state(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    source_path: Path,
) -> dict[str, Any]:
    documents = list(
        db.scalars(
            select(Document)
            .where(
                Document.knowledge_base_id == recovery.knowledge_base_id,
                Document.source_path == str(source_path),
            )
            .order_by(Document.id.asc())
            .limit(2)
            .with_for_update()
        ).all()
    )
    if len(documents) > 1:
        raise RuntimeError(f"Multiple documents claim one ingestion source path: {source_path}")
    document = documents[0] if documents else None
    active_versions: list[DocumentVersion] = []
    active_chunks: list[Chunk] = []
    if document is not None:
        active_versions = list(
            db.scalars(
                select(DocumentVersion)
                .where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.is_active.is_(True),
                )
                .order_by(DocumentVersion.id.asc())
                .with_for_update()
            ).all()
        )
        if len(active_versions) > 1:
            raise RuntimeError(
                f"Document {document.id} has multiple active versions before batch mutation"
            )
        active_chunks = list(
            db.scalars(
                select(Chunk)
                .where(Chunk.document_id == document.id, Chunk.state == "active")
                .order_by(Chunk.chunk_index.asc(), Chunk.id.asc())
                .with_for_update()
            ).all()
        )
    source_files = list(
        db.scalars(
            select(SourceFile)
            .where(
                SourceFile.knowledge_base_id == recovery.knowledge_base_id,
                SourceFile.source_path == str(source_path),
            )
            .order_by(SourceFile.id.asc())
            .with_for_update()
        ).all()
    )
    return {
        "protocol_version": INGESTION_FILE_BEFORE_PROTOCOL_VERSION,
        "batch_recovery_id": recovery.id,
        "knowledge_base_id": recovery.knowledge_base_id,
        "source_path": str(source_path),
        "document": (
            {
                "exists": True,
                "id": str(document.id),
                "metadata": _document_metadata_payload(document),
                "created_at": document.created_at.isoformat() if document.created_at else None,
                "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            }
            if document is not None
            else {"exists": False, "id": None, "metadata": None, "created_at": None, "updated_at": None}
        ),
        "active_document_version_ids": [str(row.id) for row in active_versions],
        "active_chunk_ids": [str(row.id) for row in active_chunks],
        "active_chunk_scope_hash": compute_chunk_scope_hash(active_chunks),
        "source_files": [_source_file_card(row) for row in source_files],
    }


def _prepare_file_stage(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    source_path: Path,
    sequence_index: int,
) -> IngestionFileStage:
    stage = db.scalar(
        select(IngestionFileStage)
        .where(
            IngestionFileStage.batch_recovery_id == recovery.id,
            IngestionFileStage.source_path == str(source_path),
        )
        .with_for_update()
    )
    if stage is not None:
        before = dict(stage.before_state_json or {})
        if (
            stage.sequence_index != sequence_index
            or before.get("protocol_version") != INGESTION_FILE_BEFORE_PROTOCOL_VERSION
            or before.get("batch_recovery_id") != recovery.id
            or before.get("source_path") != str(source_path)
            or _canonical_payload_hash(before) != stage.before_state_hash
        ):
            raise RuntimeError(f"Ingestion file stage {stage.id} has an invalid before-image")
        return stage
    before_state = _file_before_state(db, recovery=recovery, source_path=source_path)
    stage = IngestionFileStage(
        batch_recovery_id=recovery.id,
        knowledge_base_id=recovery.knowledge_base_id,
        source_path=str(source_path),
        sequence_index=int(sequence_index),
        status="prepared",
        phase="prepared",
        before_state_json=before_state,
        before_state_hash=_canonical_payload_hash(before_state),
        write_set_json={},
        compensation_json={},
    )
    db.add(stage)
    recovery.status = "parsing"
    db.commit()
    db.refresh(stage)
    return stage


def _file_committed_write_set(
    db: Session,
    *,
    stage: IngestionFileStage,
    job: IngestionJob,
    document: Document,
    version: DocumentVersion,
    source_file: SourceFile,
    parse_job: ParseJob,
    chunks: list[Chunk],
) -> dict[str, Any]:
    chunk_ids = [str(chunk.id) for chunk in chunks]
    vector_records = list(
        db.scalars(
            select(VectorRecord)
            .where(VectorRecord.chunk_id.in_(chunk_ids))
            .order_by(VectorRecord.collection_name.asc(), VectorRecord.qdrant_point_id.asc())
        ).all()
    ) if chunk_ids else []
    structure_node_ids = list(
        db.scalars(
            select(ChunkStructureNode.id)
            .where(ChunkStructureNode.document_version_id == version.id)
            .order_by(ChunkStructureNode.id.asc())
        ).all()
    )
    structure_edge_ids = list(
        db.scalars(
            select(ChunkStructureEdge.id)
            .where(ChunkStructureEdge.document_version_id == version.id)
            .order_by(ChunkStructureEdge.id.asc())
        ).all()
    )
    mapping_ids = list(
        db.scalars(
            select(ChunkStructureMapping.id)
            .where(ChunkStructureMapping.document_version_id == version.id)
            .order_by(ChunkStructureMapping.id.asc())
        ).all()
    )
    span_ids = list(
        db.scalars(
            select(ChunkSpan.id)
            .where(ChunkSpan.document_version_id == version.id)
            .order_by(ChunkSpan.id.asc())
        ).all()
    )
    coordinate_ids = list(
        db.scalars(
            select(ChunkCoordinate.id)
            .where(ChunkCoordinate.document_version_id == version.id)
            .order_by(ChunkCoordinate.id.asc())
        ).all()
    )
    active_scope = list(
        db.scalars(active_chunks_query(job.knowledge_base_id)).all()
    )
    vector_cards: list[dict[str, Any]] = []
    for row in vector_records:
        diagnostics = dict(row.diagnostics_json or {})
        write_intent_id = str(diagnostics.get("qdrant_write_intent_id") or "")
        write_protocol = str(
            diagnostics.get("qdrant_write_protocol_version") or ""
        )
        vector_hash_protocol = str(
            diagnostics.get("vector_payload_hash_protocol") or ""
        )
        vector_payload_hash = str(row.payload_hash or "")
        if (
            not write_intent_id
            or write_protocol != QDRANT_OUTBOX_PROTOCOL_VERSION
            or vector_hash_protocol != VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION
            or len(vector_payload_hash) != 64
        ):
            raise RuntimeError(
                f"VectorRecord {row.id} has no complete committed Qdrant write identity"
            )
        vector_cards.append(
            {
                "id": str(row.id),
                "chunk_id": str(row.chunk_id),
                "qdrant_point_id": str(row.qdrant_point_id),
                "collection_name": str(row.collection_name),
                "payload_hash": vector_payload_hash,
                "vector_payload_hash": vector_payload_hash,
                "vector_payload_hash_protocol": vector_hash_protocol,
                "qdrant_write_intent_id": write_intent_id,
                "qdrant_write_protocol_version": write_protocol,
                "vector_status": str(row.vector_status),
            }
        )
    return {
        "protocol_version": INGESTION_FILE_WRITE_SET_PROTOCOL_VERSION,
        "file_stage_id": stage.id,
        "batch_recovery_id": stage.batch_recovery_id,
        "knowledge_base_id": job.knowledge_base_id,
        "source_path": stage.source_path,
        "ingestion_job_id": job.id,
        "document_id": document.id,
        "document_version_id": version.id,
        "source_file_id": source_file.id,
        "parse_job_id": parse_job.id,
        "chunk_version": int(version.version),
        "chunk_ids": chunk_ids,
        "structure_node_ids": [str(item) for item in structure_node_ids],
        "structure_edge_ids": [str(item) for item in structure_edge_ids],
        "structure_mapping_ids": [str(item) for item in mapping_ids],
        "chunk_span_ids": [str(item) for item in span_ids],
        "chunk_coordinate_ids": [str(item) for item in coordinate_ids],
        "vector_records": vector_cards,
        "active_chunk_scope_hash_after": compute_chunk_scope_hash(active_scope),
        "committed_at": datetime.utcnow().isoformat(),
    }


def _mark_file_stage_committed(
    db: Session,
    *,
    file_stage_id: str,
    job: IngestionJob,
    document: Document,
    version: DocumentVersion,
    source_file: SourceFile,
    parse_job: ParseJob,
    chunks: list[Chunk],
) -> None:
    stage = db.scalar(
        select(IngestionFileStage)
        .where(IngestionFileStage.id == file_stage_id)
        .with_for_update()
    )
    if stage is None:
        raise RuntimeError(f"Durable ingestion file stage is missing: {file_stage_id}")
    before_state = dict(stage.before_state_json or {})
    if (
        stage.status not in {"prepared", "parsing"}
        or stage.source_path != job.source_path
        or before_state.get("knowledge_base_id") != job.knowledge_base_id
        or _canonical_payload_hash(before_state) != stage.before_state_hash
    ):
        raise RuntimeError(f"Durable ingestion file stage {stage.id} drifted before commit")
    write_set = _file_committed_write_set(
        db,
        stage=stage,
        job=job,
        document=document,
        version=version,
        source_file=source_file,
        parse_job=parse_job,
        chunks=chunks,
    )
    stage.ingestion_job_id = job.id
    stage.document_id = document.id
    stage.status = "indexed_committed"
    stage.phase = "indexed"
    stage.write_set_json = write_set
    stage.write_set_hash = _canonical_payload_hash(write_set)
    stage.error_message = None
    stage.completed_at = datetime.utcnow()
    job.stats = {
        **(job.stats or {}),
        "batch_file_stage_id": stage.id,
        "batch_file_write_set_hash": stage.write_set_hash,
    }


def create_sync_batch(db: Session, knowledge_base_id: str, root: Path, trigger_source: str = "storage") -> IngestionBatch:
    ensure_storage_durability_ready()
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise LookupError(f"Knowledge base not found: {knowledge_base_id}")
    files = [
        validate_knowledge_base_source_path(
            path,
            knowledge_base_source_root=knowledge_base.source_root,
        )
        for path in collect_source_documents(root)
    ]
    batch = IngestionBatch(
        knowledge_base_id=knowledge_base_id,
        trigger_source=trigger_source,
        source_root=str(root),
        total_files=len(files),
        status="queued",
        stats={"file_paths": [str(path) for path in files], "phase": "queued"},
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def create_uploaded_files_batch(db: Session, knowledge_base_id: str, file_paths: list[Path], *, force: bool = False, full_reparse: bool = False) -> IngestionBatch:
    ensure_storage_durability_ready()
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        raise LookupError(f"Knowledge base not found: {knowledge_base_id}")
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"]
    validated_paths = [
        validate_knowledge_base_source_path(
            path,
            knowledge_base_source_root=knowledge_base.source_root,
        )
        for path in file_paths
    ]
    for path in validated_paths:
        validate_source_content_path(
            path,
            storage_root,
            max_bytes=get_settings().upload_max_bytes,
        )
    source_root = str(validated_paths[0].parent if validated_paths else get_settings().data_root)
    batch = IngestionBatch(
        knowledge_base_id=knowledge_base_id,
        trigger_source="upload",
        source_root=source_root,
        total_files=len(validated_paths),
        status="queued",
        stats={"file_paths": [str(path) for path in validated_paths], "force": force, "full_reparse": full_reparse, "phase": "queued"},
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def create_context_graph_rebuild_batch(db: Session, knowledge_base_id: str, *, layers: list[str] | None = None) -> IngestionBatch:
    ensure_storage_durability_ready()
    batch = IngestionBatch(
        knowledge_base_id=knowledge_base_id,
        trigger_source="graph_rebuild",
        source_root=str(get_settings().data_root),
        total_files=1,
        status="queued",
        stats={
            "phase": "queued",
            "maintenance_task": "context_graph_rebuild",
            "layers": layers or ["chunk-relation", "mid-concepts", "coarse-concepts", "context-graph"],
            "parse_committed": False,
        },
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def mark_batch_task_started(batch_id: str, task_id: str | None, task_name: str) -> None:
    from app.db import SessionLocal

    with SessionLocal() as db:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            return
        if batch.status in TERMINAL_STATES:
            return
        task_ids = set((batch.stats or {}).get("batch_task_ids") or [])
        if task_id:
            task_ids.add(task_id)
        batch.worker_id = current_worker_id()
        batch.heartbeat_at = datetime.utcnow()
        batch.stats = {**(batch.stats or {}), "celery_task_id": task_id, "celery_task_name": task_name, "batch_task_ids": sorted(task_ids)}
        db.commit()


def mark_batch_worker_heartbeat(db: Session, batch: IngestionBatch, *, phase: str | None = None) -> None:
    batch.worker_id = current_worker_id()
    batch.heartbeat_at = datetime.utcnow()
    stats = dict(batch.stats or {})
    if phase:
        stats["phase"] = phase
    stats["heartbeat_at"] = batch.heartbeat_at.isoformat()
    batch.stats = stats


def get_batch_status(db: Session, batch_id: str) -> dict | None:
    batch = db.get(IngestionBatch, batch_id)
    if batch is None:
        return None
    finalize_cancelling_batch_if_released(db, batch)
    return summarize_batch(batch)


def batch_task_ids(stats: dict) -> list[str]:
    task_ids: list[str] = []
    for value in stats.get("batch_task_ids") or []:
        if value:
            task_ids.append(str(value))
    celery_task_id = stats.get("celery_task_id")
    if celery_task_id:
        task_ids.append(str(celery_task_id))
    return sorted(set(task_ids))


def _celery_control_app():
    from celery import Celery

    settings = get_settings()
    return Celery("knowledge_base_cancel_control", broker=settings.redis_url, backend=settings.redis_url)


def revoke_celery_batch_tasks(task_ids: list[str]) -> dict:
    task_ids = sorted({str(task_id) for task_id in task_ids if task_id})
    if not task_ids:
        return {"attempted": False, "task_ids": [], "reason": "no_celery_task_ids"}
    result = {
        "attempted": True,
        "terminate": True,
        "signal": CANCEL_TERMINATE_SIGNAL,
        "task_ids": task_ids,
        "revoked_task_ids": [],
        "errors": [],
    }
    try:
        app = _celery_control_app()
        for task_id in task_ids:
            app.control.revoke(task_id, terminate=True, signal=CANCEL_TERMINATE_SIGNAL)
            result["revoked_task_ids"].append(task_id)
    except Exception as exc:
        result["errors"].append(exception_message(exc))
        result["ok"] = False
    else:
        result["ok"] = True
    return result


def inspect_celery_batch_tasks(task_ids: list[str]) -> dict:
    task_ids = sorted({str(task_id) for task_id in task_ids if task_id})
    if not task_ids:
        return {"attempted": False, "task_ids": [], "active_task_ids": [], "reserved_task_ids": [], "scheduled_task_ids": []}
    result = {
        "attempted": True,
        "task_ids": task_ids,
        "active_task_ids": [],
        "reserved_task_ids": [],
        "scheduled_task_ids": [],
        "errors": [],
    }
    try:
        inspector = _celery_control_app().control.inspect(timeout=1.0)
        snapshots = {
            "active_task_ids": inspector.active() or {},
            "reserved_task_ids": inspector.reserved() or {},
            "scheduled_task_ids": inspector.scheduled() or {},
        }
        task_id_set = set(task_ids)
        for output_key, workers in snapshots.items():
            matched: set[str] = set()
            for worker_items in workers.values():
                for item in worker_items or []:
                    request_payload = item.get("request") if isinstance(item, dict) else None
                    request_id = request_payload.get("id") if isinstance(request_payload, dict) else None
                    candidate = str((item.get("id") if isinstance(item, dict) else None) or request_id or "")
                    if candidate in task_id_set:
                        matched.add(candidate)
            result[output_key] = sorted(matched)
    except Exception as exc:
        result["errors"].append(exception_message(exc))
        result["ok"] = False
    else:
        result["ok"] = True
    return result


def celery_tasks_released(inspection: dict) -> bool:
    if inspection.get("ok") is not True:
        return False
    return not (inspection.get("active_task_ids") or inspection.get("reserved_task_ids") or inspection.get("scheduled_task_ids"))


def inspect_celery_batch_task_results(task_ids: list[str]) -> dict[str, Any]:
    task_ids = sorted({str(task_id) for task_id in task_ids if task_id})
    result: dict[str, Any] = {
        "source": "celery_result_backend_async_result_v1",
        "attempted": bool(task_ids),
        "task_ids": task_ids,
        "states": {},
        "terminal_task_ids": [],
        "errors": [],
    }
    if not task_ids:
        return result
    try:
        app = _celery_control_app()
        states = {
            task_id: str(app.AsyncResult(task_id).state or "PENDING").upper()
            for task_id in task_ids
        }
        result["states"] = states
        result["terminal_task_ids"] = sorted(
            task_id for task_id, state in states.items() if state in CELERY_TERMINAL_TASK_STATES
        )
    except Exception as exc:
        result["errors"].append(exception_message(exc))
        result["ok"] = False
    else:
        result["ok"] = True
    return result


def celery_task_results_terminal(inspection: dict[str, Any]) -> bool:
    if inspection.get("ok") is not True:
        return False
    task_ids = {str(item) for item in (inspection.get("task_ids") or [])}
    terminal_ids = {str(item) for item in (inspection.get("terminal_task_ids") or [])}
    return bool(task_ids) and task_ids == terminal_ids


def mark_batch_cancelled(db: Session, batch: IngestionBatch, *, cancellation_status: str) -> None:
    stats = dict(batch.stats or {})
    batch.status = "cancelled"
    batch.completed_at = datetime.utcnow()
    batch.worker_id = None
    batch.heartbeat_at = None
    stats["cancel_requested"] = True
    stats["cancellation_status"] = cancellation_status
    stats["phase"] = "cancelled"
    stats["cancelled_at"] = batch.completed_at.isoformat()
    batch.stats = stats
    for job in batch.jobs:
        if job.status not in TERMINAL_STATES:
            job.status = "cancelled"


def _document_metadata_intent_for_recovery(job: IngestionJob) -> dict[str, Any] | None:
    stats = dict(job.stats or {})
    if DOCUMENT_METADATA_INTENT_KEY not in stats:
        return None
    raw_intent = stats.get(DOCUMENT_METADATA_INTENT_KEY)
    if not isinstance(raw_intent, dict):
        raise DocumentMetadataRestoreError(
            f"Ingestion job {job.id} has a non-object document metadata intent"
        )
    return dict(raw_intent)


def _record_metadata_intent_recovery_failure(
    db: Session,
    *,
    job_id: str,
    exc: BaseException,
) -> None:
    db.rollback()
    job = db.get(IngestionJob, job_id)
    if job is None:
        return
    message = f"metadata intent release recovery failed: {_metadata_failure_message(exc)}"[:2000]
    job.status = "failed"
    job.error_message = message
    job.stats = {
        **(job.stats or {}),
        "phase": "metadata_release_recovery_failed",
        "manual_review_required": True,
        "metadata_release_recovery_error": {
            "protocol_version": DOCUMENT_METADATA_RELEASE_RECOVERY_PROTOCOL_VERSION,
            "type": exc.__class__.__name__,
            "message": str(exc)[:2000],
            "failed_at": datetime.utcnow().isoformat(),
        },
    }
    db.commit()


@contextmanager
def _metadata_recovery_resource_fence(
    db: Session,
    *,
    knowledge_base_id: str,
    batch_id: str,
):
    recovery_owner_token = ingestion_batch_recovery_owner_token_for_batch(
        db,
        knowledge_base_id=knowledge_base_id,
        batch_id=batch_id,
    )
    requested_owner = recovery_owner_token or batch_id
    dialect_name = str(db.get_bind().dialect.name)
    if dialect_name == "sqlite":
        durable_fence = ingestion_batch_recovery_fence_diagnostics(
            db,
            knowledge_base_id=knowledge_base_id,
            operation=INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
            owner_id=requested_owner,
        )
        yield {
            "acquired": durable_fence is None,
            "backend": "sqlite_test_adapter",
            "knowledge_base_id": knowledge_base_id,
            "batch_id": batch_id,
            "recovery_owner_token": recovery_owner_token,
            **(durable_fence or {}),
        }
        return
    if dialect_name != "postgresql":
        yield {
            "acquired": False,
            "backend": dialect_name,
            "knowledge_base_id": knowledge_base_id,
            "batch_id": batch_id,
            "reason": "unsupported_database_backend",
        }
        return

    bind = db.get_bind()
    engine = bind if isinstance(bind, Engine) else bind.engine
    lock_key = advisory_lock_key(knowledge_base_resource_key(knowledge_base_id))
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
        durable_fence = (
            ingestion_batch_recovery_fence_diagnostics(
                db,
                knowledge_base_id=knowledge_base_id,
                operation=INGESTION_BATCH_RECOVERY_LOCK_OPERATION,
                owner_id=requested_owner,
            )
            if acquired
            else None
        )
        yield {
            "acquired": acquired and durable_fence is None,
            "backend": "postgresql",
            "knowledge_base_id": knowledge_base_id,
            "batch_id": batch_id,
            "recovery_owner_token": recovery_owner_token,
            "advisory_key": lock_key,
            **(
                durable_fence
                or ({} if acquired else {"reason": "knowledge_base_resource_busy"})
            ),
        }
    finally:
        try:
            if acquired:
                released = bool(
                    connection.execute(
                        text("SELECT pg_advisory_unlock(:lock_key)"),
                        {"lock_key": lock_key},
                    ).scalar_one()
                )
                connection.commit()
                if not released:
                    raise RuntimeError("PostgreSQL reported that the metadata recovery fence was not held")
        except Exception:
            connection.invalidate()
            raise
        finally:
            connection.close()


def _recover_released_batch_metadata_intents(
    db: Session,
    *,
    batch_id: str,
    reason: str,
    cancelled: bool,
) -> dict[str, Any]:
    job_ids = list(
        db.scalars(
            select(IngestionJob.id)
            .where(IngestionJob.batch_id == batch_id)
            .order_by(IngestionJob.created_at.asc(), IngestionJob.id.asc())
        ).all()
    )
    restored_job_ids: list[str] = []
    consumed_job_ids: list[str] = []
    no_intent_job_ids: list[str] = []
    failures: list[dict[str, str]] = []
    for job_id in job_ids:
        job = db.get(IngestionJob, job_id)
        if job is None:
            failures.append({"job_id": str(job_id), "message": "ingestion job disappeared during recovery"})
            continue
        try:
            intent = _document_metadata_intent_for_recovery(job)
            if intent is None:
                no_intent_job_ids.append(job.id)
                continue
            status = str(intent.get("status") or "")
            if status in {"applied", "restored", "superseded"}:
                consumed_job_ids.append(job.id)
                continue
            if status != "pending":
                raise DocumentMetadataRestoreError(
                    f"Ingestion job {job.id} has non-consumable document metadata intent status {status!r}"
                )
            recovery_exc: BaseException = (
                IngestionCancelled(reason) if cancelled else RuntimeError(reason)
            )
            _restore_pending_document_metadata_intent(
                db,
                job=job,
                intent=intent,
                exc=recovery_exc,
            )
            restored_job_ids.append(job.id)
        except Exception as exc:
            _record_metadata_intent_recovery_failure(db, job_id=job.id, exc=exc)
            failures.append({"job_id": job.id, "message": _metadata_failure_message(exc)})
    return {
        "protocol_version": DOCUMENT_METADATA_RELEASE_RECOVERY_PROTOCOL_VERSION,
        "reason": reason,
        "cancelled": cancelled,
        "job_count": len(job_ids),
        "restored_job_ids": restored_job_ids,
        "already_consumed_job_ids": consumed_job_ids,
        "no_intent_job_ids": no_intent_job_ids,
        "failures": failures,
        "ok": not failures,
        "completed_at": datetime.utcnow().isoformat(),
    }


def _mark_batch_cancel_failed_after_metadata_recovery(
    db: Session,
    *,
    batch: IngestionBatch,
    recovery: dict[str, Any],
    cancellation_status: str,
) -> None:
    now = datetime.utcnow()
    failures = list(recovery.get("failures") or [])
    message = "; ".join(str(item.get("message") or "metadata recovery failed") for item in failures)[:2000]
    batch.status = "cancel_failed"
    batch.completed_at = now
    batch.worker_id = None
    batch.heartbeat_at = None
    batch.last_error = message or "metadata intent recovery failed"
    batch.stats = {
        **(batch.stats or {}),
        "phase": "cancel_failed",
        "cancellation_status": cancellation_status,
        "cancel_failure_reason": batch.last_error,
        "manual_review_required": True,
        "metadata_intent_recovery": recovery,
        "cancel_failed_at": now.isoformat(),
    }
    for job in batch.jobs:
        if job.status not in TERMINAL_STATES:
            job.status = "failed"
            job.error_message = batch.last_error


def finalize_cancelling_batch_if_released(db: Session, batch: IngestionBatch) -> None:
    if batch.status not in CANCELLING_STATES:
        return
    stats = dict(batch.stats or {})
    task_ids = batch_task_ids(stats)
    if task_ids:
        inspection = inspect_celery_batch_tasks(task_ids)
        stats["celery_cancel_inspection"] = inspection
        batch.stats = stats
        if not celery_tasks_released(inspection):
            db.commit()
            return
    elif batch_is_worker_owned(stats):
        stats["cancellation_status"] = "worker_release_unproven"
        stats["release_proof_required"] = True
        stats["celery_cancel_inspection"] = {
            "attempted": False,
            "ok": False,
            "reason": "worker_owned_batch_has_no_task_ids",
            "task_ids": [],
        }
        batch.stats = stats
        db.commit()
        return

    with _metadata_recovery_resource_fence(
        db,
        knowledge_base_id=batch.knowledge_base_id,
        batch_id=batch.id,
    ) as resource_fence:
        batch.stats = {
            **(batch.stats or {}),
            "metadata_recovery_resource_fence": resource_fence,
        }
        if not resource_fence.get("acquired"):
            batch.stats = {
                **(batch.stats or {}),
                "metadata_recovery_deferred": {
                    "reason": "knowledge_base_resource_busy",
                    "retryable": True,
                },
            }
            db.commit()
            return
        recovery = _recover_released_batch_metadata_intents(
            db,
            batch_id=batch.id,
            reason="hard-terminated ingestion worker released after cancellation",
            cancelled=True,
        )
        batch = db.get(IngestionBatch, batch.id)
        if batch is None:
            return
        if not recovery["ok"]:
            _mark_batch_cancel_failed_after_metadata_recovery(
                db,
                batch=batch,
                recovery=recovery,
                cancellation_status="metadata_intent_recovery_failed",
            )
            db.commit()
            emit_ingestion_log(
                batch.id,
                "batch_cancel_failed",
                "Batch cancellation could not restore pending document metadata",
                state="cancel_failed",
                celery_task_ids=task_ids,
                metadata_intent_recovery=recovery,
            )
            return
        batch_recovery = db.scalar(
            select(IngestionBatchRecovery).where(
                IngestionBatchRecovery.batch_id == batch.id
            )
        )
        cancellation_compensation: dict[str, Any] = {}
        if batch_recovery is not None:
            try:
                cancellation_compensation = (
                    _restore_graph_before_scope(
                        db,
                        recovery=batch_recovery,
                        reason="hard-terminated worker released during graph phase",
                    )
                    if batch_recovery.parse_committed
                    else compensate_ingestion_batch_parse_writes(
                        db,
                        recovery=batch_recovery,
                        reason="hard-terminated worker released before parse commit boundary",
                    )
                )
            except Exception as exc:
                db.rollback()
                batch = db.get(IngestionBatch, batch.id)
                if batch is None:
                    return
                failed_recovery = {
                    "protocol_version": INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION,
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "message": exception_message(exc),
                }
                _mark_batch_cancel_failed_after_metadata_recovery(
                    db,
                    batch=batch,
                    recovery={
                        **recovery,
                        "ok": False,
                        "failures": [
                            {
                                "job_id": "batch_recovery",
                                "message": exception_message(exc),
                            }
                        ],
                    },
                    cancellation_status="batch_compensation_failed",
                )
                batch.stats = {
                    **(batch.stats or {}),
                    "batch_cancellation_compensation": failed_recovery,
                }
                db.commit()
                return
        batch = db.get(IngestionBatch, batch.id)
        if batch is None:
            return
        batch.stats = {
            **(batch.stats or {}),
            "metadata_intent_recovery": recovery,
            "batch_cancellation_compensation": cancellation_compensation,
        }
        mark_batch_cancelled(db, batch, cancellation_status="worker_released")
        db.commit()
        emit_ingestion_log(
            batch.id,
            "batch_cancelled",
            "Batch cancellation completed after worker release",
            state="cancelled",
            celery_task_ids=task_ids,
        )


def summarize_batch(batch: IngestionBatch | None) -> dict:
    if batch is None:
        return {}
    stats = dict(batch.stats or {})
    context_graph_phase = stats.get("context_graph_phase")
    current_phase = (
        stats.get("phase")
        if batch.status in TERMINAL_STATES
        else (
            f"context_graph:{context_graph_phase}"
            if context_graph_phase
            else stats.get("phase")
        )
    )
    return {
        "batch_id": batch.id,
        "knowledge_base_id": batch.knowledge_base_id,
        "state": batch.status,
        "mode": stats.get("ingestion_execution_mode"),
        "trigger_source": batch.trigger_source,
        "source_root": batch.source_root,
        "total_files": batch.total_files,
        "processed_files": batch.processed_files,
        "success_count": batch.success_count,
        "failure_count": batch.failure_count,
        "skipped_count": batch.skipped_count,
        "current_file": stats.get("current_file"),
        "current_phase": current_phase,
        "cancel_requested": bool(stats.get("cancel_requested")) or batch.status in CANCELLING_STATES,
        "last_error": batch.last_error,
        "stats": stats,
        "coverage_by_source_type": stats.get("coverage_by_source_type", {}),
        "coverage_by_language": stats.get("coverage_by_language", {}),
        "errors": [] if batch.status == "completed" else stats.get("errors", []),
        "graph_stats": stats.get("graph_stats", {}),
        "phase": stats.get("phase"),
        "parse_committed": bool(stats.get("parse_committed")),
        "batch_recovery_id": stats.get("batch_recovery_id"),
        "batch_recovery_protocol_version": stats.get(
            "batch_recovery_protocol_version"
        ),
        "v_before_batch": stats.get("v_before_batch"),
        "parse_commit_boundary": stats.get("parse_commit_boundary"),
        "cancellation_status": stats.get("cancellation_status"),
        "cancel_failure_reason": stats.get("cancel_failure_reason"),
        "manual_review_required": bool(stats.get("manual_review_required", False)),
        "celery_task_id": stats.get("celery_task_id"),
        "celery_task_name": stats.get("celery_task_name"),
        "batch_task_ids": stats.get("batch_task_ids", []),
        "batch_worker_ids": [batch.worker_id] if batch.worker_id else [],
        "worker_id": batch.worker_id,
        "heartbeat_at": batch.heartbeat_at,
        "created_at": batch.created_at,
        "started_at": batch.started_at,
        "completed_at": batch.completed_at,
    }


def request_batch_cancel_control(db: Session, batch_id: str, knowledge_base_id: str) -> dict | None:
    batch = db.get(IngestionBatch, batch_id)
    if batch is None:
        return None
    if batch.knowledge_base_id != knowledge_base_id:
        raise PermissionError("Batch belongs to another knowledge base")
    if batch.status in TERMINAL_STATES:
        return summarize_batch(batch)

    stats = dict(batch.stats or {})
    task_ids = batch_task_ids(stats)
    now = datetime.utcnow()
    stats["cancel_requested"] = True
    stats["cancel_requested_at"] = now.isoformat()
    stats["cancellation_status"] = "cancel_requested"
    if task_ids:
        revoke_result = revoke_celery_batch_tasks(task_ids)
        stats["celery_revoke"] = revoke_result
        if revoke_result.get("ok") is False:
            stats["cancel_failure_reason"] = "; ".join(revoke_result.get("errors") or ["celery revoke failed"])
            stats["manual_review_required"] = True
            stats["cancellation_status"] = "worker_terminate_failed"
        else:
            stats["cancellation_status"] = "worker_terminate_requested"
    batch.status = "cancelling" if batch_is_worker_owned(stats) or task_ids else "cancel_requested"
    batch.stats = stats
    for job in batch.jobs:
        if job.status not in TERMINAL_STATES:
            job.status = "cancel_requested"
    db.commit()
    emit_ingestion_log(
        batch.id,
        "batch_cancel_requested",
        "Batch cancellation requested; worker termination signalled" if task_ids else "Batch cancellation requested",
        state=batch.status,
        celery_task_ids=task_ids,
        cancellation_status=stats.get("cancellation_status"),
    )
    batch = db.get(IngestionBatch, batch_id)
    if batch is not None:
        finalize_cancelling_batch_if_released(db, batch)
    return summarize_batch(batch)


async def ingest_file(
    db: Session,
    source_path: Path,
    *,
    trigger_source: str = "upload",
    existing_job_id: str | None = None,
    batch_id: str | None = None,
    knowledge_base_id: str | None = None,
    rebuild_graph: bool = True,
    force: bool = False,
    target_version: int | None = None,
    file_stage_id: str | None = None,
) -> dict:
    knowledge_base = resolve_knowledge_base(db, knowledge_base_id)
    ensure_knowledge_base_storage_durability_ready(
        knowledge_base.name,
        knowledge_base_source_root=knowledge_base.source_root,
    )
    if active_ingestion_resource_lease(knowledge_base.id) is None and existing_job_id:
        waiting_job = db.get(IngestionJob, existing_job_id)
        if waiting_job is not None:
            waiting_job.stats = {
                **(waiting_job.stats or {}),
                "phase": "waiting_resource_lock",
                "ingestion_resource_lock": ingestion_resource_lock_wait_diagnostics(
                    db,
                    knowledge_base.id,
                    operation="ingest_file",
                    batch_id=batch_id,
                ),
            }
            db.commit()
    async with knowledge_base_ingestion_resource_lock(
        db,
        knowledge_base.id,
        operation="ingest_file",
        batch_id=batch_id,
    ):
        resolved_path = source_path.absolute()
        try:
            resolved_path = validate_knowledge_base_source_path(
                source_path,
                knowledge_base_source_root=knowledge_base.source_root,
            )
            return await _ingest_file_locked(
                db,
                resolved_path,
                trigger_source=trigger_source,
                existing_job_id=existing_job_id,
                batch_id=batch_id,
                knowledge_base_id=knowledge_base.id,
                rebuild_graph=rebuild_graph,
                force=force,
                target_version=target_version,
                file_stage_id=file_stage_id,
            )
        except asyncio.CancelledError as exc:
            _restore_document_metadata_after_failure(
                db,
                knowledge_base_id=knowledge_base.id,
                path=resolved_path,
                existing_job_id=existing_job_id,
                exc=exc,
            )
            raise
        except Exception as exc:
            _restore_document_metadata_after_failure(
                db,
                knowledge_base_id=knowledge_base.id,
                path=resolved_path,
                existing_job_id=existing_job_id,
                exc=exc,
            )
            raise


async def _ingest_file_locked(
    db: Session,
    source_path: Path,
    *,
    trigger_source: str = "upload",
    existing_job_id: str | None = None,
    batch_id: str | None = None,
    knowledge_base_id: str | None = None,
    rebuild_graph: bool = True,
    force: bool = False,
    target_version: int | None = None,
    file_stage_id: str | None = None,
) -> dict:
    settings = get_settings()
    knowledge_base = resolve_knowledge_base(db, knowledge_base_id)
    resource_lock = active_ingestion_resource_lease(knowledge_base.id)
    if resource_lock is None:
        raise RuntimeError("ingest_file entered its mutation phase without the knowledge-base resource lock")
    path = source_path.resolve()
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise RuntimeError(f"Unsupported file type: {path.suffix}")
    if not path.exists() or not path.is_file():
        raise RuntimeError(f"File not found: {path}")
    if target_version is not None and int(target_version) < 1:
        raise ValueError("target_version must be at least 1")
    expected_checksum = _pending_metadata_candidate_checksum_for_snapshot(
        db,
        knowledge_base_id=knowledge_base.id,
        path=path,
        existing_job_id=existing_job_id,
    )
    storage_root = settings.knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"]
    try:
        frozen_snapshot = await run_bounded_source_io(
            snapshot_source_file,
            path,
            knowledge_base_source_root=knowledge_base.source_root,
            expected_checksum=expected_checksum,
            max_bytes=settings.upload_max_bytes,
        )
    except UploadChecksumMismatchError as exc:
        raise SourceSnapshotError(
            "Source bytes changed before the immutable attempt snapshot was committed "
            f"for {path}"
        ) from exc
    storage_path = frozen_snapshot.canonical_path
    checksum = frozen_snapshot.checksum
    parsed_source_type = ""
    sections = []
    parse_error: Exception | None = None
    try:
        parsed_source_type, sections = parse_document(frozen_snapshot)
    except Exception as exc:
        parse_error = exc
    ingestion_root = settings.knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["ingestion_root"]
    await run_bounded_source_io(
        replay_frozen_source_snapshot,
        frozen_snapshot,
        authorized_root=ingestion_root,
    )
    if parse_error is None:
        refresh_runtime_settings_if_needed()
        settings = get_settings()
        if frozen_snapshot.size_bytes > int(settings.upload_max_bytes):
            raise SourceSnapshotError(
                "Frozen source snapshot exceeds the refreshed upload hard limit"
            )
    source_type = source_type_from_path(path)
    document, job, _metadata_intent = _stage_document_metadata_for_ingestion(
        db,
        knowledge_base=knowledge_base,
        path=path,
        checksum=checksum,
        source_type=source_type,
        trigger_source=trigger_source,
        batch_id=batch_id,
        existing_job_id=existing_job_id,
    )
    if parse_error is not None:
        raise parse_error
    if file_stage_id is not None:
        file_stage = db.scalar(
            select(IngestionFileStage)
            .where(IngestionFileStage.id == file_stage_id)
            .with_for_update()
        )
        if file_stage is None:
            raise RuntimeError(f"Durable ingestion file stage is missing: {file_stage_id}")
        if (
            file_stage.source_path != str(path)
            or file_stage.status != "prepared"
            or _canonical_payload_hash(dict(file_stage.before_state_json or {}))
            != file_stage.before_state_hash
        ):
            raise RuntimeError(f"Durable ingestion file stage {file_stage.id} is not commit-ready")
        file_stage.status = "parsing"
        file_stage.phase = "parsing"
        file_stage.ingestion_job_id = job.id
        file_stage.document_id = document.id
    runtime_snapshot = {
        "fixed_chunk_size_tokens": settings.fixed_chunk_size_tokens,
        "fixed_chunk_overlap_tokens": settings.fixed_chunk_overlap_tokens,
        "context_package_token_budget": settings.context_package_token_budget,
    }
    job.status = "parsing"
    job.document_id = document.id
    job.source_path = str(path)
    job.logical_source_slot_key = document.logical_source_slot_key
    job.source_slot_protocol_version = document.source_slot_protocol_version
    job.stats = {
        **(job.stats or {}),
        "phase": "parsing",
        "runtime_settings_snapshot": runtime_snapshot,
        "ingestion_resource_lock": resource_lock.diagnostics(
            reentrant=resource_lock.operation != "ingest_file"
        ),
    }
    db.flush()

    source_file = db.scalar(
        select(SourceFile).where(
            SourceFile.knowledge_base_id == knowledge_base.id,
            SourceFile.logical_source_slot_key
            == document.logical_source_slot_key,
        )
    )
    if source_file is None:
        source_file = SourceFile(
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            source_path=str(path),
            logical_source_slot_key=document.logical_source_slot_key,
            source_slot_protocol_version=document.source_slot_protocol_version,
            checksum=checksum,
            source_type=source_type,
            size_bytes=frozen_snapshot.size_bytes,
            metadata_json={},
        )
        db.add(source_file)
    else:
        source_file.document_id = document.id
        source_file.source_path = str(path)
        source_file.source_slot_protocol_version = (
            document.source_slot_protocol_version
        )
        source_file.checksum = checksum
        source_file.source_type = source_type
        source_file.size_bytes = frozen_snapshot.size_bytes
        source_file.state = "active"
    db.flush()

    existing_active = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
    chunk_version = target_version or knowledge_base.current_chunk_version or 1
    if existing_active == 0 and (knowledge_base.current_chunk_version or 0) == 0:
        chunk_version = 1
    version_number = chunk_version
    active_versions = list(
        db.scalars(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document.id,
                DocumentVersion.is_active.is_(True),
            )
        ).all()
    )
    if len(active_versions) > 1:
        raise DocumentMetadataRestoreError(
            f"Document has multiple active parse attempts before promotion: document_id={document.id}, count={len(active_versions)}"
        )
    for old_version in active_versions:
        old_version.is_active = False
    if active_versions:
        db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=version_number,
        checksum=checksum,
        storage_path=str(storage_path),
        is_active=True,
    )
    db.add(version)
    db.flush()

    parse_job = ParseJob(
        knowledge_base_id=knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        ingestion_job_id=job.id,
        source_file_id=source_file.id,
        parser_protocol_version="parser_v1",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(parse_job)
    db.flush()
    parse_job.status = "completed"
    parse_job.completed_at = datetime.utcnow()
    parse_job.stats_json = {"section_count": len(sections), "source_type": parsed_source_type}
    parse_job.diagnostics_json = {"sections": sections_to_json(sections)[:20]}
    language_identity = _resolve_and_stage_document_language_identity(
        job=job,
        document=document,
        version=version,
        source_file=source_file,
        parse_job=parse_job,
        sections=sections,
        knowledge_base_id=knowledge_base.id,
        path=path,
        checksum=checksum,
        source_type=source_type,
    )
    db.flush()

    job.status = "chunking"
    job.stats = {
        **(job.stats or {}),
        "phase": "chunking",
        "section_count": len(sections),
        "runtime_settings_snapshot": {
            **runtime_snapshot,
            "fixed_chunk_size_tokens": settings.fixed_chunk_size_tokens,
            "fixed_chunk_overlap_tokens": settings.fixed_chunk_overlap_tokens,
        },
    }
    chunks = write_chunks_and_structure(
        db,
        knowledge_base=knowledge_base,
        document=document,
        version=version,
        sections=sections,
        chunk_version=chunk_version,
        chunk_size=settings.fixed_chunk_size_tokens,
        chunk_overlap=settings.fixed_chunk_overlap_tokens,
    )
    db.flush()

    refresh_runtime_settings_if_needed()
    job.status = "embedding"
    job.stats = {**(job.stats or {}), "phase": "embedding", "chunk_count": len(chunks)}
    local_hints = build_local_context_hints(db, chunks)
    index_stats = await write_contextual_indexes(
        db,
        knowledge_base=knowledge_base,
        chunks=chunks,
        local_hints=local_hints,
        ingestion_job_id=job.id,
    )
    job.status = "completed"
    job.stats = {**(job.stats or {}), "phase": "indexed", "index_stats": index_stats, "chunk_count": len(chunks)}
    db.flush()
    if rebuild_graph:
        refresh_runtime_settings_if_needed()
        await rebuild_context_graph(db, knowledge_base.id, batch_id=batch_id, chunk_version_incremented=chunk_version > (knowledge_base.current_chunk_version or 0))
    if knowledge_base.current_chunk_version < chunk_version:
        knowledge_base.current_chunk_version = chunk_version
    _mark_document_metadata_intent_applied(db, job=job, document=document)
    if file_stage_id is not None:
        _mark_file_stage_committed(
            db,
            file_stage_id=file_stage_id,
            job=job,
            document=document,
            version=version,
            source_file=source_file,
            parse_job=parse_job,
            chunks=chunks,
        )
    db.commit()
    if batch_id:
        emit_ingestion_log(batch_id, "file_indexed", f"{path.name} indexed into {len(chunks)} chunks", source_path=str(path), chunk_count=len(chunks), **index_stats)
    return {
        "status": "completed",
        "document_id": document.id,
        "chunk_count": len(chunks),
        "source_type": source_type,
        "language": language_identity["language"],
        "language_detection_hash": language_identity["language_detection_hash"],
        "stats": {
            "chunks": len(chunks),
            "language_identity": language_identity_summary(language_identity),
            **index_stats,
        },
    }


def _validated_file_stage(
    db: Session,
    stage: IngestionFileStage,
    *,
    recovery: IngestionBatchRecovery | None = None,
    require_write_set: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with db.no_autoflush:
        parent = recovery or db.get(IngestionBatchRecovery, stage.batch_recovery_id)
    if (
        parent is None
        or stage.batch_recovery_id != parent.id
        or stage.knowledge_base_id != parent.knowledge_base_id
    ):
        raise RuntimeError(
            f"Ingestion file stage {stage.id} recovery/knowledge-base ownership is invalid"
        )
    expected_phase = INGESTION_FILE_STAGE_STATUS_PHASES.get(str(stage.status))
    if expected_phase is None or stage.phase != expected_phase:
        raise RuntimeError(
            f"Ingestion file stage {stage.id} status/phase state is invalid"
        )
    before_state = dict(stage.before_state_json or {})
    if (
        before_state.get("protocol_version") != INGESTION_FILE_BEFORE_PROTOCOL_VERSION
        or before_state.get("batch_recovery_id") != stage.batch_recovery_id
        or before_state.get("knowledge_base_id") != stage.knowledge_base_id
        or before_state.get("source_path") != stage.source_path
        or _canonical_payload_hash(before_state) != stage.before_state_hash
    ):
        raise RuntimeError(f"Ingestion file stage {stage.id} before-image is invalid")
    write_set = dict(stage.write_set_json or {})
    if require_write_set:
        if (
            write_set.get("protocol_version") != INGESTION_FILE_WRITE_SET_PROTOCOL_VERSION
            or write_set.get("file_stage_id") != stage.id
            or write_set.get("batch_recovery_id") != stage.batch_recovery_id
            or write_set.get("knowledge_base_id") != stage.knowledge_base_id
            or write_set.get("source_path") != stage.source_path
            or not stage.write_set_hash
            or _canonical_payload_hash(write_set) != stage.write_set_hash
        ):
            raise RuntimeError(f"Ingestion file stage {stage.id} committed write set is invalid")
    elif write_set or stage.write_set_hash:
        raise RuntimeError(f"Uncommitted ingestion file stage {stage.id} unexpectedly has a write set")
    with db.no_autoflush:
        job = db.get(IngestionJob, stage.ingestion_job_id) if stage.ingestion_job_id else None
        document = db.get(Document, stage.document_id) if stage.document_id else None
    if stage.ingestion_job_id and (
        job is None
        or job.knowledge_base_id != stage.knowledge_base_id
        or job.batch_id != parent.batch_id
    ):
        raise RuntimeError(f"Ingestion file stage {stage.id} ingestion-job owner is invalid")
    if stage.document_id and (
        document is None or document.knowledge_base_id != stage.knowledge_base_id
    ):
        raise RuntimeError(f"Ingestion file stage {stage.id} document owner is invalid")
    if require_write_set and (
        write_set.get("ingestion_job_id") != stage.ingestion_job_id
        or write_set.get("document_id") != stage.document_id
    ):
        raise RuntimeError(f"Ingestion file stage {stage.id} write-set owner is invalid")
    return before_state, write_set


def _mark_file_stage_failed(
    db: Session,
    *,
    stage_id: str,
    exc: BaseException,
    cancelled: bool,
) -> None:
    stage = db.get(IngestionFileStage, stage_id)
    if stage is None:
        raise RuntimeError(f"Durable ingestion file stage disappeared: {stage_id}")
    _validated_file_stage(db, stage, require_write_set=False)
    stage.status = "cancel_observed" if cancelled else "failed"
    stage.phase = "cancel_observed" if cancelled else "failed"
    stage.error_message = _metadata_failure_message(exc)
    stage.completed_at = datetime.utcnow()
    db.commit()


class IngestionCompensationOwnershipDrift(RuntimeError):
    pass


def _qdrant_delete_source_binding(
    *,
    recovery: IngestionBatchRecovery,
    stage: IngestionFileStage,
) -> dict[str, str]:
    write_set_hash = str(stage.write_set_hash or "").lower()
    if len(write_set_hash) != 64 or any(
        character not in "0123456789abcdef" for character in write_set_hash
    ):
        raise IngestionCompensationOwnershipDrift(
            f"File stage {stage.id} has no valid frozen write-set hash"
        )
    return {
        "protocol_version": QDRANT_DELETE_SOURCE_BINDING_PROTOCOL_VERSION,
        "recovery_id": str(recovery.id),
        "file_stage_id": str(stage.id),
        "write_set_hash": write_set_hash,
    }


def _mark_qdrant_compensation_manual_review(
    db: Session,
    *,
    recovery_id: str,
    stage_id: str,
    error: BaseException,
) -> None:
    db.rollback()
    recovery = db.get(IngestionBatchRecovery, recovery_id, with_for_update=True)
    stage = db.get(IngestionFileStage, stage_id, with_for_update=True)
    if recovery is None or stage is None:
        raise RuntimeError(
            "Ingestion compensation ownership drift could not be durably recorded"
        ) from error
    diagnostic = {
        "protocol_version": "ingestion_qdrant_compensation_manual_review_v1",
        "reason": "qdrant_owner_or_payload_hash_drift",
        "recovery_id": recovery.id,
        "file_stage_id": stage.id,
        "write_set_hash": stage.write_set_hash,
        "error_type": error.__class__.__name__,
        "error": exception_message(error),
        "observed_at": datetime.utcnow().isoformat(),
        "current_owner_captured_as_before_image": False,
        "automatic_delete_forbidden": True,
    }
    stage.status = "manual_review"
    stage.phase = "manual_review"
    stage.compensation_json = {
        **dict(stage.compensation_json or {}),
        "manual_review": diagnostic,
    }
    recovery.status = "manual_review"
    recovery.diagnostics_json = {
        **dict(recovery.diagnostics_json or {}),
        "qdrant_compensation_manual_review": diagnostic,
    }
    db.commit()


def _validate_frozen_vector_collection_ownership(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    stage: IngestionFileStage,
    collection_name: str,
    cards: list[dict[str, Any]],
    qdrant_points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_point_ids = [str(card.get("qdrant_point_id") or "") for card in cards]
    observed_by_id = {
        str(point.get("id") or ""): dict(point) for point in qdrant_points
    }
    if (
        not all(expected_point_ids)
        or len(expected_point_ids) != len(set(expected_point_ids))
        or not all(observed_by_id)
        or len(observed_by_id) != len(qdrant_points)
        or set(observed_by_id) != set(expected_point_ids)
    ):
        raise IngestionCompensationOwnershipDrift(
            f"File stage {stage.id} Qdrant point scope is missing, duplicated, or expanded"
        )
    ordered_points: list[dict[str, Any]] = []
    for card in cards:
        record_id = str(card.get("id") or "")
        point_id = str(card.get("qdrant_point_id") or "")
        chunk_id = str(card.get("chunk_id") or "")
        frozen_owner = str(card.get("qdrant_write_intent_id") or "")
        frozen_write_protocol = str(
            card.get("qdrant_write_protocol_version") or ""
        )
        frozen_hash_protocol = str(
            card.get("vector_payload_hash_protocol") or ""
        )
        frozen_payload_hash = str(card.get("vector_payload_hash") or "")
        record = db.get(VectorRecord, record_id)
        diagnostics = dict(record.diagnostics_json or {}) if record is not None else {}
        if (
            record is None
            or record.knowledge_base_id != recovery.knowledge_base_id
            or str(record.chunk_id) != chunk_id
            or str(record.qdrant_point_id) != point_id
            or str(record.collection_name) != collection_name
            or str(record.vector_status) != str(card.get("vector_status") or "")
            or str(record.payload_hash) != frozen_payload_hash
            or diagnostics.get("qdrant_write_intent_id") != frozen_owner
            or diagnostics.get("qdrant_write_protocol_version")
            != frozen_write_protocol
            or diagnostics.get("vector_payload_hash_protocol")
            != frozen_hash_protocol
            or frozen_write_protocol != QDRANT_OUTBOX_PROTOCOL_VERSION
            or frozen_hash_protocol != VECTOR_PAYLOAD_HASH_PROTOCOL_VERSION
        ):
            raise IngestionCompensationOwnershipDrift(
                f"File stage {stage.id} PostgreSQL VectorRecord ownership drifted for {point_id}"
            )
        try:
            committed_target = validated_committed_qdrant_upsert_target(
                db,
                intent_id=frozen_owner,
                knowledge_base_id=recovery.knowledge_base_id,
                collection_name=collection_name,
                point_id=point_id,
            )
        except Exception as exc:
            raise IngestionCompensationOwnershipDrift(
                f"File stage {stage.id} committed Qdrant upsert owner is invalid for {point_id}"
            ) from exc
        durable_payload = dict(committed_target["target_payload"])
        qdrant_point = observed_by_id[point_id]
        qdrant_payload = dict(qdrant_point.get("payload") or {})
        expected_owner_fields = {
            "knowledge_base_id": recovery.knowledge_base_id,
            "chunk_id": chunk_id,
            "qdrant_write_intent_id": frozen_owner,
            "qdrant_write_protocol_version": frozen_write_protocol,
            "vector_payload_hash_protocol": frozen_hash_protocol,
            "vector_payload_hash": frozen_payload_hash,
        }
        if any(
            durable_payload.get(field) != expected
            for field, expected in expected_owner_fields.items()
        ):
            raise IngestionCompensationOwnershipDrift(
                f"File stage {stage.id} upsert outbox owner/hash drifted for {point_id}"
            )
        if any(
            qdrant_payload.get(field) != expected
            for field, expected in expected_owner_fields.items()
        ):
            raise IngestionCompensationOwnershipDrift(
                f"File stage {stage.id} current Qdrant owner/hash drifted for {point_id}"
            )
        ordered_points.append(qdrant_point)
    return ordered_points


def _stage_qdrant_delete_intents(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    stage: IngestionFileStage,
    knowledge_base: KnowledgeBase,
    write_set: dict[str, Any],
) -> list[str]:
    records = [dict(item) for item in (write_set.get("vector_records") or [])]
    by_collection: dict[str, list[dict[str, Any]]] = {}
    for card in records:
        collection_name = str(card.get("collection_name") or "")
        point_id = str(card.get("qdrant_point_id") or "")
        if (
            not collection_name
            or not point_id
            or not card.get("qdrant_write_intent_id")
            or not card.get("qdrant_write_protocol_version")
            or not card.get("vector_payload_hash_protocol")
            or not card.get("vector_payload_hash")
        ):
            raise RuntimeError(f"File stage {stage.id} contains an invalid vector write card")
        by_collection.setdefault(collection_name, []).append(card)
    source_binding = _qdrant_delete_source_binding(recovery=recovery, stage=stage)
    compensation = dict(stage.compensation_json or {})
    intent_map = dict(compensation.get("qdrant_delete_intents") or {})
    expected_collections = sorted(by_collection)
    unknown = sorted(set(intent_map) - set(expected_collections))
    if unknown:
        raise RuntimeError(
            f"File stage {stage.id} has out-of-scope Qdrant compensation intents: {unknown}"
        )
    for collection_name in expected_collections:
        cards = by_collection[collection_name]
        point_ids = [str(card["qdrant_point_id"]) for card in cards]
        if len(point_ids) != len(set(point_ids)):
            raise RuntimeError(f"File stage {stage.id} repeats a Qdrant point id")
        intent_id = str(intent_map.get(collection_name) or "")
        row = db.get(IngestionCompensationLog, intent_id) if intent_id else None
        try:
            if intent_id and row is None:
                raise IngestionCompensationOwnershipDrift(
                    f"File stage {stage.id} references a missing Qdrant delete intent"
                )
            if row is None:
                row = find_qdrant_delete_intent_by_source_binding(
                    db,
                    knowledge_base_id=recovery.knowledge_base_id,
                    collection_name=collection_name,
                    target_ids=point_ids,
                    recovery_id=recovery.id,
                    file_stage_id=stage.id,
                    write_set_hash=str(stage.write_set_hash or ""),
                )
            if row is not None:
                validated_delete = validated_qdrant_delete_intent(row)
                if (
                    validated_delete.get("source_binding") != source_binding
                    or validated_delete.get("collection_name") != collection_name
                    or validated_delete.get("target_ids") != point_ids
                    or validated_delete.get("reason")
                    != QDRANT_INGESTION_PARSE_COMPENSATION_REASON
                ):
                    raise IngestionCompensationOwnershipDrift(
                        f"File stage {stage.id} Qdrant delete intent binding drifted"
                    )
                # The independently committed before-image is the Qdrant side
                # of the three-way fence once an intent already exists.
                _validate_frozen_vector_collection_ownership(
                    db,
                    recovery=recovery,
                    stage=stage,
                    collection_name=collection_name,
                    cards=cards,
                    qdrant_points=list(validated_delete.get("before_points") or []),
                )
                intent_id = str(row.id)
            else:
                # Validate PostgreSQL + committed upsert owner before observing
                # Qdrant.  The current point is never persisted as a delete
                # before-image until all three stores agree with the frozen
                # file write set.
                store = VectorStore(knowledge_base.name, collection_name=collection_name)
                current_points = store.get_points(point_ids)
                current_points = _validate_frozen_vector_collection_ownership(
                    db,
                    recovery=recovery,
                    stage=stage,
                    collection_name=collection_name,
                    cards=cards,
                    qdrant_points=current_points,
                )
                intent_id = persist_qdrant_delete_attempt(
                    db,
                    knowledge_base_id=recovery.knowledge_base_id,
                    collection_name=collection_name,
                    target_ids=point_ids,
                    current_points=current_points,
                    reason=QDRANT_INGESTION_PARSE_COMPENSATION_REASON,
                    source_binding=source_binding,
                )
                row = db.get(IngestionCompensationLog, intent_id)
                if row is None:
                    raise RuntimeError(
                        "Durable Qdrant delete intent disappeared after persistence"
                    )
                validated_delete = validated_qdrant_delete_intent(row)
                if validated_delete.get("source_binding") != source_binding:
                    raise RuntimeError(
                        "Durable Qdrant delete intent lost its recovery source binding"
                    )
        except IngestionCompensationOwnershipDrift as exc:
            _mark_qdrant_compensation_manual_review(
                db,
                recovery_id=recovery.id,
                stage_id=stage.id,
                error=exc,
            )
            raise RuntimeError(
                f"File stage {stage.id} requires manual Qdrant compensation review"
            ) from exc
        except Exception as exc:
            # Invalid durable outbox contracts are ownership drift; transport
            # failures from the Qdrant read/persist path remain retryable.
            if row is not None:
                drift = IngestionCompensationOwnershipDrift(str(exc))
                _mark_qdrant_compensation_manual_review(
                    db,
                    recovery_id=recovery.id,
                    stage_id=stage.id,
                    error=drift,
                )
                raise RuntimeError(
                    f"File stage {stage.id} requires manual Qdrant compensation review"
                ) from exc
            raise
        if intent_map.get(collection_name) != intent_id:
            intent_map[collection_name] = intent_id
            compensation = {
                **compensation,
                "protocol_version": INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION,
                "qdrant_delete_intents": dict(sorted(intent_map.items())),
                "qdrant_source_binding": source_binding,
                "qdrant_intents_persisted_at": datetime.utcnow().isoformat(),
            }
            stage = db.get(IngestionFileStage, stage.id, with_for_update=True)
            if stage is None:
                raise RuntimeError("File stage disappeared after Qdrant intent persistence")
            stage.compensation_json = compensation
            db.commit()
            stage = db.get(IngestionFileStage, stage.id)
            if stage is None:
                raise RuntimeError("File stage disappeared after Qdrant intent binding")
        row = db.get(IngestionCompensationLog, intent_id)
        if row is None:
            raise RuntimeError(f"File stage {stage.id} Qdrant delete intent is missing")
        if row.status != "committed":
            store = VectorStore(knowledge_base.name, collection_name=collection_name)
            replay_qdrant_delete_intent(db, store=store, row=row)
        stage = db.get(IngestionFileStage, stage.id)
        if stage is None:
            raise RuntimeError("File stage disappeared after Qdrant compensation")
        stage.compensation_json = {
            **dict(stage.compensation_json or {}),
            "qdrant_delete_intents": dict(sorted(intent_map.items())),
            "qdrant_external_applied_at": datetime.utcnow().isoformat(),
        }
        db.commit()
    return [str(intent_map[key]) for key in sorted(intent_map)]


def _restore_source_files(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    stage: IngestionFileStage,
    before_state: dict[str, Any],
    write_set: dict[str, Any],
) -> None:
    before_cards = [dict(item) for item in (before_state.get("source_files") or [])]
    before_by_id = {str(card.get("id") or ""): card for card in before_cards}
    if not all(before_by_id) or len(before_by_id) != len(before_cards):
        raise RuntimeError(f"File stage {stage.id} source-file before-image is invalid")
    for source_file_id, card in before_by_id.items():
        row = db.get(SourceFile, source_file_id)
        if row is None or row.knowledge_base_id != recovery.knowledge_base_id:
            raise RuntimeError(f"File stage {stage.id} cannot restore SourceFile {source_file_id}")
        row.document_id = card.get("document_id")
        row.source_path = str(card["source_path"])
        row.checksum = str(card["checksum"])
        row.source_type = str(card["source_type"])
        row.size_bytes = int(card.get("size_bytes") or 0)
        row.metadata_json = dict(card.get("metadata_json") or {})
        row.state = str(card["state"])
        if card.get("created_at"):
            row.created_at = datetime.fromisoformat(str(card["created_at"]))
        if card.get("updated_at"):
            row.updated_at = datetime.fromisoformat(str(card["updated_at"]))
    candidate_source_file_id = str(write_set.get("source_file_id") or "")
    if candidate_source_file_id and candidate_source_file_id not in before_by_id:
        row = db.get(SourceFile, candidate_source_file_id)
        if row is None or row.knowledge_base_id != recovery.knowledge_base_id:
            raise RuntimeError(f"File stage {stage.id} candidate SourceFile is missing")
        row.state = "inactive"


def _restore_file_stage_database_scope(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    stage: IngestionFileStage,
    delete_intent_ids: list[str],
) -> None:
    before_state, write_set = _validated_file_stage(
        db,
        stage,
        recovery=recovery,
        require_write_set=True,
    )
    if write_set.get("knowledge_base_id") != recovery.knowledge_base_id:
        raise RuntimeError(f"File stage {stage.id} write set belongs to another knowledge base")
    candidate_chunk_ids = [str(item) for item in (write_set.get("chunk_ids") or [])]
    candidate_version_id = str(write_set.get("document_version_id") or "")
    candidate_document_id = str(write_set.get("document_id") or "")
    if not candidate_version_id or not candidate_document_id:
        raise RuntimeError(f"File stage {stage.id} write set has no candidate version identity")
    candidate_chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.id.in_(candidate_chunk_ids))
            .order_by(Chunk.id.asc())
            .with_for_update()
        ).all()
    ) if candidate_chunk_ids else []
    if sorted(str(row.id) for row in candidate_chunks) != sorted(candidate_chunk_ids):
        raise RuntimeError(f"File stage {stage.id} candidate chunk write set is incomplete")
    for row in candidate_chunks:
        if row.knowledge_base_id != recovery.knowledge_base_id:
            raise RuntimeError(f"File stage {stage.id} candidate chunk crossed knowledge bases")
        row.state = "inactive"
    candidate_version = db.get(DocumentVersion, candidate_version_id, with_for_update=True)
    if candidate_version is None or candidate_version.document_id != candidate_document_id:
        raise RuntimeError(f"File stage {stage.id} candidate DocumentVersion is missing")
    candidate_version.is_active = False
    db.flush()
    before_version_ids = [str(item) for item in (before_state.get("active_document_version_ids") or [])]
    for version_id in before_version_ids:
        row = db.get(DocumentVersion, version_id, with_for_update=True)
        if row is None or row.document_id != candidate_document_id:
            raise RuntimeError(f"File stage {stage.id} cannot restore DocumentVersion {version_id}")
        row.is_active = True
    before_chunk_ids = [str(item) for item in (before_state.get("active_chunk_ids") or [])]
    before_chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.id.in_(before_chunk_ids))
            .order_by(Chunk.id.asc())
            .with_for_update()
        ).all()
    ) if before_chunk_ids else []
    if sorted(str(row.id) for row in before_chunks) != sorted(before_chunk_ids):
        raise RuntimeError(f"File stage {stage.id} old active chunk scope is incomplete")
    for row in before_chunks:
        if row.document_id != candidate_document_id:
            raise RuntimeError(f"File stage {stage.id} old chunk ownership drifted")
        row.state = "active"
    document_before = dict(before_state.get("document") or {})
    document = db.get(Document, candidate_document_id, with_for_update=True)
    if document is None or document.knowledge_base_id != recovery.knowledge_base_id:
        raise RuntimeError(f"File stage {stage.id} candidate Document is missing")
    if document_before.get("exists") is True:
        if document_before.get("id") != document.id:
            raise RuntimeError(f"File stage {stage.id} Document before-image identity drifted")
        _apply_document_metadata(document, dict(document_before.get("metadata") or {}))
        if document_before.get("created_at"):
            document.created_at = datetime.fromisoformat(str(document_before["created_at"]))
        if document_before.get("updated_at"):
            document.updated_at = datetime.fromisoformat(str(document_before["updated_at"]))
    else:
        document.is_active = False
    _restore_source_files(
        db,
        recovery=recovery,
        stage=stage,
        before_state=before_state,
        write_set=write_set,
    )
    vector_record_ids = [str(card.get("id") or "") for card in (write_set.get("vector_records") or [])]
    vector_records = list(
        db.scalars(
            select(VectorRecord)
            .where(VectorRecord.id.in_(vector_record_ids))
            .order_by(VectorRecord.id.asc())
            .with_for_update()
        ).all()
    ) if vector_record_ids else []
    if sorted(str(row.id) for row in vector_records) != sorted(vector_record_ids):
        raise RuntimeError(f"File stage {stage.id} vector write set is incomplete")
    for row in vector_records:
        row.vector_status = "compensated"
        row.diagnostics_json = {
            **dict(row.diagnostics_json or {}),
            "batch_cancel_compensation": {
                "protocol_version": INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION,
                "batch_recovery_id": recovery.id,
                "file_stage_id": stage.id,
                "qdrant_delete_intent_ids": list(delete_intent_ids),
            },
        }
    job = db.get(IngestionJob, write_set.get("ingestion_job_id"), with_for_update=True)
    if job is None or job.knowledge_base_id != recovery.knowledge_base_id:
        raise RuntimeError(f"File stage {stage.id} ingestion job is missing")
    intent = dict((job.stats or {}).get(DOCUMENT_METADATA_INTENT_KEY) or {})
    if intent.get("status") not in {"applied", "batch_compensated"}:
        raise RuntimeError(
            f"File stage {stage.id} committed metadata intent is not compensation-safe"
        )
    job.status = "cancelled"
    job.error_message = "batch cancelled before parse commit boundary"
    job.stats = {
        **(job.stats or {}),
        "phase": "batch_parse_compensated",
        DOCUMENT_METADATA_INTENT_KEY: {
            **intent,
            "status": "batch_compensated",
            "batch_recovery_id": recovery.id,
            "file_stage_id": stage.id,
            "compensated_at": datetime.utcnow().isoformat(),
        },
    }
    mark_qdrant_delete_attempts_committed(db, intent_ids=delete_intent_ids)
    stage.status = "compensated"
    stage.phase = "compensated"
    stage.compensation_json = {
        **dict(stage.compensation_json or {}),
        "database_scope_restored_at": datetime.utcnow().isoformat(),
        "restored_active_document_version_ids": before_version_ids,
        "restored_active_chunk_ids": before_chunk_ids,
        "candidate_chunk_ids_deactivated": candidate_chunk_ids,
        "qdrant_delete_intent_ids": list(delete_intent_ids),
        "version_restore_source": "file_before_state",
        "version_minus_one_inference_used": False,
    }
    stage.completed_at = datetime.utcnow()


def _restore_chunk_version_descriptors(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    before_state: dict[str, Any],
) -> None:
    cards = [dict(item) for item in (before_state.get("chunk_version_descriptors") or [])]
    before_ids = {str(card.get("id") or "") for card in cards}
    if "" in before_ids or len(before_ids) != len(cards):
        raise RuntimeError(f"Batch recovery {recovery.id} ChunkVersion before-image is invalid")
    for card in cards:
        row = db.get(ChunkVersion, card["id"], with_for_update=True)
        if row is None or row.knowledge_base_id != recovery.knowledge_base_id:
            raise RuntimeError(f"Batch recovery {recovery.id} cannot restore ChunkVersion {card['id']}")
        row.chunk_schema_version = str(card["chunk_schema_version"])
        row.tokenizer_version = str(card["tokenizer_version"])
        row.chunk_size = int(card["chunk_size"])
        row.chunk_overlap = int(card["chunk_overlap"])
        row.state_hash = str(card["state_hash"])
        row.stats_json = dict(card.get("stats_json") or {})
        row.diagnostics_json = dict(card.get("diagnostics_json") or {})
        row.state = str(card["state"])
        if card.get("created_at"):
            row.created_at = datetime.fromisoformat(str(card["created_at"]))
    candidate_descriptors = list(
        db.scalars(
            select(ChunkVersion)
            .where(
                ChunkVersion.knowledge_base_id == recovery.knowledge_base_id,
                ChunkVersion.chunk_version == recovery.target_version,
            )
            .with_for_update()
        ).all()
    )
    for row in candidate_descriptors:
        if row.id not in before_ids:
            row.state = "inactive"


def _dispatch_batch_recovery_cache_invalidation(
    db: Session,
    *,
    recovery_id: str,
) -> dict[str, Any]:
    recovery = db.get(IngestionBatchRecovery, recovery_id, with_for_update=True)
    if recovery is None:
        raise RuntimeError(f"Batch recovery disappeared before cache invalidation: {recovery_id}")
    compensation = dict(recovery.compensation_json or {})
    cache_state = dict(compensation.get("cache_invalidation") or {})
    if cache_state.get("status") == "dispatched":
        return cache_state
    attempt_count = int(cache_state.get("attempt_count") or 0) + 1
    cache_state = {
        **cache_state,
        "protocol_version": "ingestion_batch_cache_invalidation_v1",
        "status": "pending",
        "attempt_count": attempt_count,
        "knowledge_base_id": recovery.knowledge_base_id,
        "last_attempt_at": datetime.utcnow().isoformat(),
    }
    recovery.compensation_json = {
        **compensation,
        "cache_invalidation": cache_state,
    }
    db.commit()
    try:
        result = invalidate_context_graph_cache_after_commit(
            recovery.knowledge_base_id,
            strict=True,
        )
        if result is not True:
            raise RuntimeError(
                "Strict knowledge-base cache invalidation did not report complete success "
                f"(receipt_type={type(result).__name__})"
            )
    except Exception as exc:
        db.rollback()
        recovery = db.get(IngestionBatchRecovery, recovery_id, with_for_update=True)
        if recovery is None:
            raise RuntimeError("Batch recovery disappeared after cache invalidation failure") from exc
        compensation = dict(recovery.compensation_json or {})
        cache_state = dict(compensation.get("cache_invalidation") or {})
        cache_state.update(
            {
                "status": "pending",
                "retryable": True,
                "last_error_type": exc.__class__.__name__,
                "last_error": exception_message(exc),
                "failed_at": datetime.utcnow().isoformat(),
            }
        )
        recovery.compensation_json = {
            **compensation,
            "cache_invalidation": cache_state,
        }
        db.commit()
        return cache_state
    recovery = db.get(IngestionBatchRecovery, recovery_id, with_for_update=True)
    if recovery is None:
        raise RuntimeError("Batch recovery disappeared after cache invalidation dispatch")
    compensation = dict(recovery.compensation_json or {})
    cache_state = dict(compensation.get("cache_invalidation") or {})
    cache_state.update(
        {
            "status": "dispatched",
            "retryable": False,
            "dispatched_at": datetime.utcnow().isoformat(),
            "result": {"ok": True, "strict": True},
        }
    )
    cache_state.pop("last_error", None)
    cache_state.pop("last_error_type", None)
    recovery.compensation_json = {
        **compensation,
        "cache_invalidation": cache_state,
    }
    db.commit()
    return cache_state


def _versioned_graph_commit_cache_invalidation_state(
    *,
    knowledge_base_id: str,
    attempt_id: str,
    plan_hash: str,
    context_graph_state_id: str,
    graph_write_set_hash: str,
) -> dict[str, Any]:
    binding = {
        "protocol_version": "versioned_graph_commit_cache_binding_v1",
        "attempt_id": str(attempt_id),
        "plan_hash": str(plan_hash),
        "context_graph_state_id": str(context_graph_state_id),
        "graph_write_set_hash": str(graph_write_set_hash),
    }
    return {
        "protocol_version": "ingestion_batch_cache_invalidation_v1",
        "status": "pending",
        "attempt_count": 0,
        "knowledge_base_id": str(knowledge_base_id),
        "reason": "versioned_graph_commit",
        "graph_commit_binding": binding,
        "graph_commit_binding_hash": _canonical_payload_hash(binding),
    }


def _completed_versioned_graph_retry_stats(
    existing: dict[str, Any] | None,
    *,
    context_state: ContextGraphState,
    graph_write_set_hash: str,
    attempt_id: str,
    plan_hash: str,
) -> dict[str, Any]:
    stats = dict(existing or {})
    prior_failure = stats.pop("versioned_graph_retry_failure", None)
    failure_history = list(stats.get("versioned_graph_retry_failure_history") or [])
    if isinstance(prior_failure, dict):
        failure_history.append(dict(prior_failure))
    if failure_history:
        stats["versioned_graph_retry_failure_history"] = failure_history
    prior_errors = list(stats.pop("errors", []) or [])
    if prior_errors:
        stats["ingestion_error_history"] = [
            *list(stats.get("ingestion_error_history") or []),
            *prior_errors,
        ]
    for key in (
        "context_graph_phase",
        "context_graph_heartbeat_at",
        "context_graph_metrics",
        "current_file",
        "failed_at",
        "failure_compensation",
        "interrupted_recovery_deferred",
        "metadata_recovery_resource_fence",
    ):
        stats.pop(key, None)
    stats.update(
        {
            "phase": "completed",
            "parse_committed": True,
            "graph_stats": dict(context_state.stats_json or {}),
            "graph_write_set_hash": str(graph_write_set_hash),
            "active_versioned_graph_retry": None,
            "versioned_graph_retry_completed": {
                "protocol_version": VERSIONED_GRAPH_RETRY_PROTOCOL_VERSION,
                "attempt_id": str(attempt_id),
                "plan_hash": str(plan_hash),
                "context_graph_state_id": str(context_state.id),
                "gray_zone_model_call_count": 0,
            },
        }
    )
    return stats


def reconcile_pending_ingestion_recovery_cache_invalidations(db: Session) -> dict[str, Any]:
    recovery_ids = list(
        db.scalars(
            select(IngestionBatchRecovery.id)
            .where(
                IngestionBatchRecovery.status.in_(
                    ["parse_compensated", "graph_compensated"]
                )
            )
            .order_by(IngestionBatchRecovery.updated_at.asc(), IngestionBatchRecovery.id.asc())
        ).all()
    )
    attempted: list[str] = []
    pending: list[str] = []
    for recovery_id in recovery_ids:
        recovery = db.get(IngestionBatchRecovery, recovery_id)
        cache_state = dict((recovery.compensation_json or {}).get("cache_invalidation") or {}) if recovery else {}
        if cache_state.get("status") == "dispatched":
            continue
        attempted.append(str(recovery_id))
        result = _dispatch_batch_recovery_cache_invalidation(
            db,
            recovery_id=str(recovery_id),
        )
        if result.get("status") != "dispatched":
            pending.append(str(recovery_id))
    return {
        "protocol_version": "ingestion_batch_cache_invalidation_reconcile_v1",
        "attempted_recovery_ids": attempted,
        "pending_recovery_ids": pending,
        "ok": not pending,
    }


def reconcile_pending_ingestion_batch_recoveries(db: Session) -> dict[str, Any]:
    """Retry terminal ``cancel_failed`` batches whose durable scope is pending.

    The ordinary interrupted-batch loop owns non-terminal rows.  This pass is
    intentionally limited to terminal failure states so an external Qdrant or
    database compensation failure cannot strand a recovery merely because the
    first worker already recorded ``cancel_failed``.  Cache dispatch has its
    own terminal-state reconciliation pass below.
    """

    recovery_ids = list(
        db.scalars(
            select(IngestionBatchRecovery.id)
            .where(
                IngestionBatchRecovery.status.in_(
                    [
                        "prepared",
                        "parsing",
                        "parse_compensation_pending",
                        "parse_compensating",
                        "graph_building",
                        "graph_compensation_pending",
                    ]
                )
            )
            .order_by(IngestionBatchRecovery.updated_at.asc(), IngestionBatchRecovery.id.asc())
        ).all()
    )
    attempted: list[str] = []
    recovered: list[str] = []
    pending: list[str] = []
    for recovery_id in recovery_ids:
        recovery = db.get(IngestionBatchRecovery, recovery_id)
        batch = db.get(IngestionBatch, recovery.batch_id) if recovery is not None else None
        if recovery is None or batch is None or batch.status not in {"cancel_failed", "failed"}:
            continue
        attempted.append(str(recovery_id))
        with _metadata_recovery_resource_fence(
            db,
            knowledge_base_id=recovery.knowledge_base_id,
            batch_id=batch.id,
        ) as resource_fence:
            if not resource_fence.get("acquired"):
                pending.append(str(recovery_id))
                continue
            metadata_recovery = _recover_released_batch_metadata_intents(
                db,
                batch_id=batch.id,
                reason="retry durable terminal batch recovery",
                cancelled=bool((batch.stats or {}).get("cancel_requested")),
            )
            if not metadata_recovery["ok"]:
                pending.append(str(recovery_id))
                continue
            try:
                compensation = (
                    _restore_graph_before_scope(
                        db,
                        recovery=recovery,
                        reason="retry graph compensation after terminal failure",
                    )
                    if recovery.parse_committed
                    else compensate_ingestion_batch_parse_writes(
                        db,
                        recovery=recovery,
                        reason="retry parse compensation after terminal failure",
                    )
                )
            except Exception as exc:
                db.rollback()
                batch = db.get(IngestionBatch, batch.id)
                if batch is not None:
                    batch.stats = {
                        **(batch.stats or {}),
                        "pending_batch_recovery_retry": {
                            "protocol_version": INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION,
                            "recovery_id": str(recovery_id),
                            "retryable": True,
                            "last_error_type": exc.__class__.__name__,
                            "last_error": exception_message(exc),
                            "attempted_at": datetime.utcnow().isoformat(),
                        },
                    }
                    db.commit()
                pending.append(str(recovery_id))
                continue
            batch = db.get(IngestionBatch, batch.id)
            if batch is None:
                pending.append(str(recovery_id))
                continue
            cancellation_requested = bool((batch.stats or {}).get("cancel_requested"))
            batch.stats = {
                **(batch.stats or {}),
                "metadata_intent_recovery": metadata_recovery,
                "pending_batch_recovery_reconciled": compensation,
                "manual_review_required": False,
            }
            if cancellation_requested:
                mark_batch_cancelled(
                    db,
                    batch,
                    cancellation_status="durable_compensation_retried",
                )
            else:
                batch.status = "failed"
                batch.worker_id = None
                batch.heartbeat_at = None
                batch.completed_at = datetime.utcnow()
                batch.stats = {
                    **(batch.stats or {}),
                    "phase": "interrupted_recovered",
                    "retryable": True,
                }
            db.commit()
            recovered.append(str(recovery_id))
    return {
        "protocol_version": "ingestion_batch_pending_recovery_reconcile_v1",
        "attempted_recovery_ids": attempted,
        "recovered_recovery_ids": recovered,
        "pending_recovery_ids": pending,
        "ok": not pending,
    }


def compensate_ingestion_batch_parse_writes(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    reason: str,
) -> dict[str, Any]:
    before_state, _graph_before = _validate_batch_recovery(db, recovery)
    if recovery.parse_committed:
        raise RuntimeError("Parse writes cannot be rolled back after the graph-build boundary")
    knowledge_base = db.get(KnowledgeBase, recovery.knowledge_base_id, with_for_update=True)
    if knowledge_base is None:
        raise RuntimeError(f"Batch recovery {recovery.id} knowledge base is missing")
    recovery.status = "parse_compensation_pending"
    recovery.compensation_json = {
        **dict(recovery.compensation_json or {}),
        "reason": str(reason),
        "parse_compensation_requested_at": datetime.utcnow().isoformat(),
    }
    db.commit()
    stages = list(
        db.scalars(
            select(IngestionFileStage)
            .where(IngestionFileStage.batch_recovery_id == recovery.id)
            .order_by(IngestionFileStage.sequence_index.desc())
        ).all()
    )
    compensated_ids: list[str] = []
    for stage in stages:
        if stage.status in {"prepared", "failed", "cancel_observed"}:
            _validated_file_stage(
                db,
                stage,
                recovery=recovery,
                require_write_set=False,
            )
            continue
        if stage.status == "compensated":
            _validated_file_stage(
                db,
                stage,
                recovery=recovery,
                require_write_set=True,
            )
            compensated_ids.append(stage.id)
            continue
        if stage.status not in {
            "indexed_committed",
            "compensation_pending",
            "compensating",
        }:
            raise RuntimeError(
                f"File stage {stage.id} cannot be parse-compensated from {stage.status}"
            )
        _before, write_set = _validated_file_stage(
            db,
            stage,
            recovery=recovery,
            require_write_set=True,
        )
        stage.status = "compensation_pending"
        stage.phase = "qdrant_compensation"
        recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
        if recovery is None:
            raise RuntimeError("Batch recovery disappeared during compensation")
        recovery.status = "parse_compensating"
        db.commit()
        stage = db.get(IngestionFileStage, stage.id)
        recovery = db.get(IngestionBatchRecovery, recovery.id)
        if stage is None or recovery is None:
            raise RuntimeError("Batch recovery scope disappeared during compensation")
        delete_intent_ids = _stage_qdrant_delete_intents(
            db,
            recovery=recovery,
            stage=stage,
            knowledge_base=knowledge_base,
            write_set=write_set,
        )
        stage = db.get(IngestionFileStage, stage.id, with_for_update=True)
        recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
        if stage is None or recovery is None:
            raise RuntimeError("Batch recovery scope disappeared before database restore")
        stage.status = "compensating"
        stage.phase = "database_restore"
        _restore_file_stage_database_scope(
            db,
            recovery=recovery,
            stage=stage,
            delete_intent_ids=delete_intent_ids,
        )
        db.commit()
        compensated_ids.append(stage.id)
    recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
    knowledge_base = db.get(KnowledgeBase, recovery.knowledge_base_id, with_for_update=True) if recovery else None
    if recovery is None or knowledge_base is None:
        raise RuntimeError("Batch recovery scope disappeared before final verification")
    before_state, _graph_before = _validate_batch_recovery(db, recovery)
    _restore_chunk_version_descriptors(db, recovery=recovery, before_state=before_state)
    knowledge_base.current_chunk_version = int(recovery.v_before_batch)
    db.flush()
    active_chunks = list(
        db.scalars(
            active_chunks_query(recovery.knowledge_base_id)
            .order_by(Chunk.document_id.asc(), Chunk.chunk_index.asc(), Chunk.id.asc())
            .execution_options(populate_existing=True)
        ).all()
    )
    active_chunk_ids = [str(row.id) for row in active_chunks]
    active_scope_hash = compute_chunk_scope_hash(active_chunks)
    active_version_ids = sorted(
        str(item)
        for item in db.scalars(
            select(DocumentVersion.id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.knowledge_base_id == recovery.knowledge_base_id,
                DocumentVersion.is_active.is_(True),
            )
        ).all()
    )
    expected_chunk_ids = [str(item) for item in (before_state.get("active_chunk_ids") or [])]
    expected_version_ids = sorted(
        str(item) for item in (before_state.get("active_document_version_ids") or [])
    )
    verification = {
        "knowledge_base_chunk_version": int(knowledge_base.current_chunk_version or 0),
        "expected_v_before_batch": int(recovery.v_before_batch),
        "active_chunk_scope_hash": active_scope_hash,
        "expected_active_chunk_scope_hash": before_state.get("active_chunk_scope_hash"),
        "active_chunk_ids_match": active_chunk_ids == expected_chunk_ids,
        "active_document_version_ids_match": active_version_ids == expected_version_ids,
        "version_minus_one_inference_used": False,
    }
    verification["ok"] = bool(
        verification["knowledge_base_chunk_version"]
        == verification["expected_v_before_batch"]
        and verification["active_chunk_scope_hash"]
        == verification["expected_active_chunk_scope_hash"]
        and verification["active_chunk_ids_match"]
        and verification["active_document_version_ids_match"]
    )
    if not verification["ok"]:
        raise RuntimeError(f"Batch parse compensation verification failed: {verification}")
    recovery.status = "parse_compensated"
    recovery.compensation_json = {
        **dict(recovery.compensation_json or {}),
        "protocol_version": INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION,
        "compensated_file_stage_ids": sorted(compensated_ids),
        "verification": verification,
        "parse_compensated_at": datetime.utcnow().isoformat(),
        "cache_invalidation": dict(
            (recovery.compensation_json or {}).get("cache_invalidation")
            or {
                "protocol_version": "ingestion_batch_cache_invalidation_v1",
                "status": "pending",
                "attempt_count": 0,
                "knowledge_base_id": recovery.knowledge_base_id,
            }
        ),
    }
    recovery.completed_at = datetime.utcnow()
    db.commit()
    _dispatch_batch_recovery_cache_invalidation(db, recovery_id=recovery.id)
    recovery = db.get(IngestionBatchRecovery, recovery.id)
    return dict(recovery.compensation_json or {}) if recovery is not None else {}


def _restore_graph_before_scope(
    db: Session,
    *,
    recovery: IngestionBatchRecovery,
    reason: str,
) -> dict[str, Any]:
    _before, graph_before = _validate_batch_recovery(db, recovery)
    if not recovery.parse_committed:
        raise RuntimeError("Graph compensation requires the durable parse-commit boundary")
    recovery.status = "graph_compensation_pending"
    db.commit()
    states = dict(graph_before.get("states") or {})
    restored: dict[str, list[str]] = {}
    for key, model in (
        ("relation", ChunkRelationGraphState),
        ("mid", MidConceptState),
        ("coarse", CoarseConceptState),
        ("context", ContextGraphState),
    ):
        cards = [dict(item) for item in (states.get(key) or [])]
        before_ids = {str(card.get("id") or "") for card in cards}
        if "" in before_ids or len(before_ids) != len(cards):
            raise RuntimeError(f"Graph before-state {key} identity is invalid")
        rows = list(
            db.scalars(
                select(model)
                .where(model.knowledge_base_id == recovery.knowledge_base_id)
                .with_for_update()
            ).all()
        )
        rows_by_id = {str(row.id): row for row in rows}
        for card in cards:
            row = rows_by_id.get(str(card["id"]))
            if row is None or _graph_state_hash(row) != str(card.get("state_hash") or ""):
                raise RuntimeError(f"Cannot restore exact {key} graph state {card.get('id')}")
        for row in rows:
            if row.state == "active" or str(row.id) in before_ids:
                row.state = "active" if str(row.id) in before_ids else "inactive"
        restored[key] = sorted(before_ids)
    pointer_card = graph_before.get("vector_runtime_pointer")
    if pointer_card is not None:
        pointer_card = dict(pointer_card)
        pointer = db.get(
            KnowledgeBaseVectorRuntimeState,
            pointer_card.get("id"),
            with_for_update=True,
        )
        if pointer is None or pointer.knowledge_base_id != recovery.knowledge_base_id:
            raise RuntimeError("Graph compensation cannot restore the vector runtime pointer")
        pointer.active_context_graph_state_id = pointer_card.get("active_context_graph_state_id")
        pointer.active_chunk_relation_graph_state_id = pointer_card.get(
            "active_chunk_relation_graph_state_id"
        )
        pointer.active_mid_concept_state_id = pointer_card.get("active_mid_concept_state_id")
        pointer.active_coarse_concept_state_id = pointer_card.get("active_coarse_concept_state_id")
        pointer.state_hash = str(pointer_card.get("state_hash") or "")
        pointer.promotion_audit_json = dict(pointer_card.get("promotion_audit_json") or {})
    recovery.status = "graph_compensated"
    recovery.compensation_json = {
        **dict(recovery.compensation_json or {}),
        "protocol_version": INGESTION_BATCH_RECOVERY_PROTOCOL_VERSION,
        "reason": str(reason),
        "parse_scope_preserved": True,
        "restored_graph_state_ids": restored,
        "graph_compensated_at": datetime.utcnow().isoformat(),
        "gray_zone_model_call_count": 0,
        "cache_invalidation": dict(
            (recovery.compensation_json or {}).get("cache_invalidation")
            or {
                "protocol_version": "ingestion_batch_cache_invalidation_v1",
                "status": "pending",
                "attempt_count": 0,
                "knowledge_base_id": recovery.knowledge_base_id,
            }
        ),
    }
    recovery.completed_at = datetime.utcnow()
    db.commit()
    _dispatch_batch_recovery_cache_invalidation(db, recovery_id=recovery.id)
    recovery = db.get(IngestionBatchRecovery, recovery.id)
    return dict(recovery.compensation_json or {}) if recovery is not None else {}


def _graph_committed_write_set(context_state: ContextGraphState) -> dict[str, Any]:
    return {
        "protocol_version": INGESTION_GRAPH_WRITE_SET_PROTOCOL_VERSION,
        "context_graph_state_id": context_state.id,
        "chunk_relation_graph_state_id": context_state.chunk_relation_graph_state_id,
        "mid_concept_state_id": context_state.mid_concept_state_id,
        "coarse_concept_state_id": context_state.coarse_concept_state_id,
        "context_graph_hash": context_state.context_graph_hash,
        "committed_at": datetime.utcnow().isoformat(),
    }


def finalize_interrupted_batches() -> None:
    from app.db import SessionLocal

    with SessionLocal() as db:
        batch_ids = list(
            db.scalars(
                select(IngestionBatch.id).where(IngestionBatch.status.notin_(TERMINAL_STATES))
            ).all()
        )
        for batch_id in batch_ids:
            batch = db.get(IngestionBatch, batch_id)
            if batch is None or batch.status in TERMINAL_STATES:
                continue
            stats = dict(batch.stats or {})
            cancellation_requested = (
                batch.status in CANCELLING_STATES
                or bool(stats.get("cancel_requested"))
            )
            if batch_is_worker_owned(stats):
                stats["api_restart_observed_at"] = datetime.utcnow().isoformat()
                task_ids = batch_task_ids(stats)
                if not task_ids:
                    stats["interrupted_recovery_deferred"] = {
                        "reason": "worker_release_unproven_missing_task_ids",
                        "release_proof_required": True,
                    }
                    batch.stats = stats
                    db.commit()
                    continue
                inspection = inspect_celery_batch_tasks(task_ids)
                stats["celery_restart_inspection"] = inspection
                if not celery_tasks_released(inspection):
                    stats["interrupted_recovery_deferred"] = {
                        "reason": "worker_release_not_proven",
                        "release_proof_required": True,
                        "inspection_ok": inspection.get("ok"),
                        "active_task_ids": list(inspection.get("active_task_ids") or []),
                        "reserved_task_ids": list(inspection.get("reserved_task_ids") or []),
                        "scheduled_task_ids": list(inspection.get("scheduled_task_ids") or []),
                    }
                    batch.stats = stats
                    db.commit()
                    continue
                task_result_inspection = inspect_celery_batch_task_results(task_ids)
                stats["celery_restart_task_results"] = task_result_inspection
                revoke = dict(stats.get("celery_revoke") or {})
                revoked_task_ids = {
                    str(task_id) for task_id in (revoke.get("revoked_task_ids") or [])
                }
                cancellation_revoke_proven = bool(
                    cancellation_requested
                    and revoke.get("ok") is True
                    and set(task_ids).issubset(revoked_task_ids)
                )
                terminal_result_proven = celery_task_results_terminal(task_result_inspection)
                if not (cancellation_revoke_proven or terminal_result_proven):
                    stats["interrupted_recovery_deferred"] = {
                        "reason": "worker_terminal_state_not_proven",
                        "release_proof_required": True,
                        "inspection_ok": inspection.get("ok"),
                        "task_result_inspection_ok": task_result_inspection.get("ok"),
                        "task_states": dict(task_result_inspection.get("states") or {}),
                        "cancellation_revoke_proven": cancellation_revoke_proven,
                    }
                    batch.stats = stats
                    db.commit()
                    continue
                batch.stats = stats
                recovery_reason = "API restart observed that the ingestion worker task was released"
            else:
                recovery_reason = "API restart terminated the in-process ingestion task"

            with _metadata_recovery_resource_fence(
                db,
                knowledge_base_id=batch.knowledge_base_id,
                batch_id=batch.id,
            ) as resource_fence:
                batch.stats = {
                    **(batch.stats or {}),
                    "metadata_recovery_resource_fence": resource_fence,
                }
                if not resource_fence.get("acquired"):
                    batch.stats = {
                        **(batch.stats or {}),
                        "interrupted_recovery_deferred": {
                            "reason": "knowledge_base_resource_busy",
                            "release_proof_required": False,
                            "retryable": True,
                        },
                    }
                    db.commit()
                    continue
                recovery = _recover_released_batch_metadata_intents(
                    db,
                    batch_id=batch.id,
                    reason=recovery_reason,
                    cancelled=cancellation_requested,
                )
                batch = db.get(IngestionBatch, batch.id)
                if batch is None:
                    continue
                if not recovery["ok"]:
                    _mark_batch_cancel_failed_after_metadata_recovery(
                        db,
                        batch=batch,
                        recovery=recovery,
                        cancellation_status="interrupted_metadata_intent_recovery_failed",
                    )
                    db.commit()
                    emit_ingestion_log(
                        batch.id,
                        "batch_cancel_failed",
                        "Interrupted batch could not restore pending document metadata",
                        state="cancel_failed",
                        metadata_intent_recovery=recovery,
                    )
                    continue
                batch_recovery = db.scalar(
                    select(IngestionBatchRecovery).where(
                        IngestionBatchRecovery.batch_id == batch.id
                    )
                )
                recovery_compensation: dict[str, Any] = {}
                if batch_recovery is not None:
                    try:
                        recovery_compensation = (
                            _restore_graph_before_scope(
                                db,
                                recovery=batch_recovery,
                                reason=f"{recovery_reason}; graph transaction interrupted",
                            )
                            if batch_recovery.parse_committed
                            else compensate_ingestion_batch_parse_writes(
                                db,
                                recovery=batch_recovery,
                                reason=f"{recovery_reason}; parse transaction interrupted",
                            )
                        )
                    except Exception as exc:
                        db.rollback()
                        batch = db.get(IngestionBatch, batch.id)
                        if batch is None:
                            continue
                        batch.status = "cancel_failed"
                        batch.completed_at = datetime.utcnow()
                        batch.worker_id = None
                        batch.heartbeat_at = None
                        batch.last_error = exception_message(exc)
                        batch.stats = {
                            **(batch.stats or {}),
                            "phase": "interrupted_recovery_failed",
                            "manual_review_required": True,
                            "cancel_failure_reason": batch.last_error,
                            "metadata_intent_recovery": recovery,
                            "batch_recovery_error_type": exc.__class__.__name__,
                        }
                        db.commit()
                        continue
                batch = db.get(IngestionBatch, batch.id)
                if batch is None:
                    continue
                batch.stats = {
                    **(batch.stats or {}),
                    "metadata_intent_recovery": recovery,
                    "batch_restart_compensation": recovery_compensation,
                }
                if cancellation_requested:
                    mark_batch_cancelled(
                        db,
                        batch,
                        cancellation_status="worker_released_after_restart",
                    )
                else:
                    batch.status = "failed"
                    batch.completed_at = datetime.utcnow()
                    batch.worker_id = None
                    batch.heartbeat_at = None
                    batch.last_error = "API restarted after the ingestion task was released"
                    batch.stats = {
                        **(batch.stats or {}),
                        "manual_review_required": False,
                        "phase": "interrupted",
                        "retryable": True,
                        "interrupted_at": batch.completed_at.isoformat(),
                    }
                    for job in batch.jobs:
                        if job.status not in TERMINAL_STATES:
                            job.status = "failed"
                            job.error_message = batch.last_error
                db.commit()

        # A failed external delete can make the first recovery attempt terminal
        # while the durable recovery row is still pending.  Retry that exact
        # write set under the same knowledge-base fence before dispatching any
        # derived cache invalidation.
        reconcile_pending_ingestion_batch_recoveries(db)

        # Cache invalidation is a separate, retryable derived side effect.  A
        # batch may already be terminal while its Redis dispatch is still
        # pending, so reconcile it outside the non-terminal batch loop.
        reconcile_pending_ingestion_recovery_cache_invalidations(db)


def batch_is_worker_owned(stats: dict) -> bool:
    return stats.get("ingestion_execution_mode") == "celery" or bool(stats.get("celery_task_id"))



async def run_batch_ingestion(batch_id: str) -> dict:
    from app.db import SessionLocal

    with SessionLocal() as db:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        file_paths = [Path(path) for path in (batch.stats or {}).get("file_paths", [])]
    return await run_uploaded_files_ingestion(batch_id, [str(path) for path in file_paths], execution_mode="celery")


def _record_batch_resource_lock_conflict(
    db: Session,
    batch: IngestionBatch,
    exc: IngestionResourceBusyError,
) -> None:
    message = exception_message(exc)
    now = datetime.utcnow()
    stats = dict(batch.stats or {})
    errors = list(stats.get("errors") or [])
    errors.append({"phase": "resource_lock", "message": message})
    batch.status = "failed"
    batch.last_error = message
    batch.completed_at = now
    batch.worker_id = None
    batch.heartbeat_at = None
    batch.stats = {
        **stats,
        "phase": "resource_lock_wait_timeout",
        "errors": errors,
        "ingestion_resource_lock": dict(exc.diagnostics),
        "retryable": True,
        "retry_guidance": "retry_after_active_ingest_or_graph_rebuild_completes",
        "failed_at": now.isoformat(),
    }
    db.commit()
    emit_ingestion_log(
        batch.id,
        "batch_resource_lock_timeout",
        message,
        error=message,
        resource_lock=exc.diagnostics,
    )


def _mark_batch_waiting_for_resource_lock(
    db: Session,
    batch: IngestionBatch,
    *,
    operation: str,
) -> None:
    now = datetime.utcnow()
    lock_diagnostics = ingestion_resource_lock_wait_diagnostics(
        db,
        batch.knowledge_base_id,
        operation=operation,
        batch_id=batch.id,
    )
    batch.worker_id = current_worker_id()
    batch.heartbeat_at = now
    batch.stats = {
        **(batch.stats or {}),
        "phase": "waiting_resource_lock",
        "ingestion_resource_lock": lock_diagnostics,
        "heartbeat_at": now.isoformat(),
    }
    db.commit()
    emit_ingestion_log(
        batch.id,
        "batch_waiting_for_resource_lock",
        "Waiting for another ingest or graph rebuild on the same knowledge base to finish",
        resource_lock=lock_diagnostics,
    )


async def run_context_graph_rebuild_batch(batch_id: str, *, execution_mode: str = "inline") -> dict:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        if batch.status in TERMINAL_STATES:
            return summarize_batch(batch)
        _mark_batch_waiting_for_resource_lock(
            db,
            batch,
            operation="context_graph_rebuild_batch",
        )
        async with knowledge_base_ingestion_resource_lock(
            db,
            batch.knowledge_base_id,
            operation="context_graph_rebuild_batch",
            batch_id=batch.id,
        ):
            return await _run_context_graph_rebuild_batch_locked(
                batch_id,
                execution_mode=execution_mode,
            )
    except IngestionResourceBusyError as exc:
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            _record_batch_resource_lock_conflict(db, batch, exc)
        raise
    finally:
        db.close()


async def _run_context_graph_rebuild_batch_locked(batch_id: str, *, execution_mode: str = "inline") -> dict:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        if batch.status in TERMINAL_STATES:
            return summarize_batch(batch)
        ensure_not_cancelled(db, batch_id)
        knowledge_base = resolve_knowledge_base(db, batch.knowledge_base_id)
        resource_lock = active_ingestion_resource_lease(knowledge_base.id)
        if resource_lock is None:
            raise RuntimeError("Context graph rebuild batch entered its mutation phase without the knowledge-base resource lock")
        recovery = _prepare_batch_recovery(
            db,
            batch=batch,
            knowledge_base=knowledge_base,
            target_version=int(knowledge_base.current_chunk_version or 0),
            full_reparse=False,
        )
        recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
        batch = db.get(IngestionBatch, batch_id)
        if recovery is None or batch is None:
            raise RuntimeError("Context graph rebuild recovery scope disappeared")
        recovery.parse_committed = True
        recovery.status = "graph_building"
        recovery.diagnostics_json = {
            **dict(recovery.diagnostics_json or {}),
            "maintenance_graph_rebuild": True,
            "parse_scope_mutated": False,
            "parse_commit_boundary": "preexisting_active_chunks",
        }
        batch.status = "extracting_graph"
        batch.started_at = datetime.utcnow()
        batch.completed_at = None
        batch.total_files = 1
        batch.processed_files = 0
        batch.success_count = 0
        batch.failure_count = 0
        batch.skipped_count = 0
        batch.worker_id = current_worker_id()
        batch.heartbeat_at = datetime.utcnow()
        batch.stats = {
            **(batch.stats or {}),
            "phase": "context_graph",
            "ingestion_execution_mode": execution_mode,
            "maintenance_task": "context_graph_rebuild",
            "batch_recovery_id": recovery.id,
            "batch_recovery_protocol_version": recovery.protocol_version,
            "parse_committed": True,
            "ingestion_resource_lock": resource_lock.diagnostics(),
            "errors": [],
        }
        db.commit()
        emit_ingestion_log(batch.id, "batch_started", "Context graph rebuild started", maintenance_task="context_graph_rebuild")
        ensure_not_cancelled(db, batch_id)
        emit_ingestion_log(batch.id, "context_graph_started", "Building four-layer context graph")
        context_state = await rebuild_context_graph(db, knowledge_base.id, batch_id=batch.id, chunk_version_incremented=False)
        graph_stats = dict(context_state.stats_json or {})
        batch = db.get(IngestionBatch, batch_id)
        recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
        if batch is None or recovery is None:
            raise RuntimeError(f"Batch {batch_id} disappeared")
        graph_write_set = _graph_committed_write_set(context_state)
        recovery.graph_write_set_json = graph_write_set
        recovery.graph_write_set_hash = _canonical_payload_hash(graph_write_set)
        recovery.status = "completed"
        recovery.completed_at = datetime.utcnow()
        batch.status = "completed"
        batch.processed_files = 1
        batch.success_count = 1
        batch.failure_count = 0
        batch.completed_at = datetime.utcnow()
        batch.worker_id = None
        batch.heartbeat_at = None
        batch.stats = {
            **(batch.stats or {}),
            "phase": "completed",
            "graph_stats": graph_stats,
            "parse_committed": False,
            "graph_rebuild_committed": True,
            "context_graph_state_id": context_state.id,
            "graph_write_set_hash": recovery.graph_write_set_hash,
        }
        db.commit()
        emit_ingestion_log(batch.id, "context_graph_completed", "Four-layer context graph is active", **graph_stats)
        emit_ingestion_log(batch.id, "batch_completed", "Context graph rebuild completed", **graph_stats)
        return summarize_batch(batch)
    except IngestionCancelled:
        db.rollback()
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            recovery = db.scalar(
                select(IngestionBatchRecovery).where(
                    IngestionBatchRecovery.batch_id == batch_id
                )
            )
            try:
                compensation = (
                    _restore_graph_before_scope(
                        db,
                        recovery=recovery,
                        reason="cooperative cancellation during maintenance graph rebuild",
                    )
                    if recovery is not None
                    else {}
                )
            except Exception as compensation_exc:
                db.rollback()
                batch = db.get(IngestionBatch, batch_id)
                if batch is not None:
                    batch.status = "cancel_failed"
                    batch.last_error = exception_message(compensation_exc)
                    batch.completed_at = datetime.utcnow()
                    batch.stats = {
                        **(batch.stats or {}),
                        "phase": "cancel_failed",
                        "cancel_requested": True,
                        "cancellation_status": "durable_compensation_pending_retry",
                        "manual_review_required": True,
                        "cancel_failure_reason": batch.last_error,
                    }
                    db.commit()
                raise
            batch = db.get(IngestionBatch, batch_id)
            if batch is None:
                raise RuntimeError("Context graph batch disappeared after compensation")
            batch.stats = {
                **(batch.stats or {}),
                "cancellation_compensation": compensation,
            }
            mark_batch_cancelled(
                db,
                batch,
                cancellation_status="graph_cancelled_parse_scope_preserved",
            )
            db.commit()
            emit_ingestion_log(
                batch.id,
                "batch_cancelled",
                "Context graph rebuild cancelled; committed chunks were preserved",
                state="cancelled",
            )
            return summarize_batch(batch)
        raise
    except Exception as exc:
        db.rollback()
        message = exception_message(exc)
        failure_audit = graph_build_failure_audit(exc)
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            recovery = db.scalar(
                select(IngestionBatchRecovery).where(
                    IngestionBatchRecovery.batch_id == batch_id
                )
            )
            compensation: dict[str, Any] = {}
            compensation_error: str | None = None
            if recovery is not None and recovery.status not in {"completed", "graph_compensated"}:
                try:
                    compensation = _restore_graph_before_scope(
                        db,
                        recovery=recovery,
                        reason=f"maintenance graph rebuild failed: {message}",
                    )
                except Exception as recovery_exc:
                    db.rollback()
                    compensation_error = exception_message(recovery_exc)
            batch = db.get(IngestionBatch, batch_id)
            if batch is None:
                raise RuntimeError("Context graph batch disappeared during failure compensation")
            stats = dict(batch.stats or {})
            errors = list(stats.get("errors") or [])
            errors.append({"phase": stats.get("phase") or batch.status, "message": message})
            batch.status = "cancel_failed" if compensation_error else "failed"
            batch.last_error = compensation_error or message
            batch.completed_at = datetime.utcnow()
            batch.worker_id = None
            batch.heartbeat_at = None
            batch.stats = {
                **stats,
                "phase": "failed",
                "errors": errors,
                "manual_review_required": bool(compensation_error),
                "failure_compensation": compensation,
                "graph_failure_audit": failure_audit,
                **({"compensation_failure": compensation_error} if compensation_error else {}),
                "failed_at": datetime.utcnow().isoformat(),
            }
            db.commit()
            emit_ingestion_log(batch.id, "batch_failed", f"Context graph rebuild failed: {message}", error=message)
        raise
    finally:
        db.close()


async def run_versioned_graph_retry(
    *,
    batch_id: str,
    expected_plan_hash: str,
    execution_mode: str = "script",
) -> dict[str, Any]:
    """Retry only the graph half of one compensated version increment."""

    from app.db import SessionLocal

    admission = SessionLocal()
    try:
        admitted_plan = plan_versioned_graph_retry(admission, batch_id=batch_id)
        if str(expected_plan_hash).lower() != str(
            admitted_plan["plan_hash"]
        ).lower():
            raise RuntimeError("Versioned graph retry plan hash is stale")
    finally:
        admission.rollback()
        admission.close()

    outer = SessionLocal()
    try:
        async with knowledge_base_ingestion_resource_lock(
            outer,
            str(admitted_plan["knowledge_base_id"]),
            operation="versioned_graph_retry",
            batch_id=batch_id,
        ) as resource_lock:
            db = SessionLocal()
            lifecycle_started = False
            graph_commit_completed = False
            plan: dict[str, Any] | None = None
            try:
                plan = plan_versioned_graph_retry(db, batch_id=batch_id)
                if plan != admitted_plan:
                    raise RuntimeError(
                        "Versioned graph retry plan drifted after resource-lock admission"
                    )
                contextual_preflight = (
                    await verify_contextual_indexes_for_graph_only_retry(
                        db,
                        str(plan["knowledge_base_id"]),
                    )
                )
                qdrant_preflight = dict(
                    contextual_preflight.get("qdrant_freshness") or {}
                )
                preflight_card = {
                    "repair_mode": contextual_preflight.get("repair_mode"),
                    "checked_chunks": contextual_preflight.get("checked_chunks"),
                    "reindexed_chunks": contextual_preflight.get(
                        "reindexed_chunks"
                    ),
                    "contextual_index_hash": contextual_preflight.get(
                        "contextual_index_hash"
                    ),
                    "qdrant_verified": qdrant_preflight.get("verified"),
                    "qdrant_verification_hash": qdrant_preflight.get(
                        "verification_hash"
                    ),
                }
                post_preflight_plan = plan_versioned_graph_retry(
                    db,
                    batch_id=batch_id,
                )
                if post_preflight_plan != plan:
                    raise RuntimeError(
                        "Versioned graph retry plan drifted during contextual preflight"
                    )
                batch = db.scalar(
                    select(IngestionBatch)
                    .where(IngestionBatch.id == batch_id)
                    .with_for_update()
                )
                recovery = db.scalar(
                    select(IngestionBatchRecovery)
                    .where(
                        IngestionBatchRecovery.id == plan["recovery_id"]
                    )
                    .with_for_update()
                )
                if batch is None or recovery is None:
                    raise RuntimeError("Versioned graph retry target disappeared")
                attempt_id = str(uuid4())
                history = list(
                    (recovery.diagnostics_json or {}).get(
                        "versioned_graph_retry_history"
                    )
                    or []
                )
                history.append(
                    {
                        "attempt_id": attempt_id,
                        "plan_hash": plan["plan_hash"],
                        "graph_failure_audit_hash": plan[
                            "graph_failure_audit_hash"
                        ],
                        "contextual_preflight_hash": _canonical_payload_hash(
                            preflight_card
                        ),
                        "status": "graph_building",
                        "started_at": datetime.utcnow().isoformat(),
                    }
                )
                recovery.status = "graph_building"
                recovery.completed_at = None
                recovery.diagnostics_json = {
                    **dict(recovery.diagnostics_json or {}),
                    "versioned_graph_retry_history": history,
                    "active_versioned_graph_retry": {
                        "protocol_version": VERSIONED_GRAPH_RETRY_PROTOCOL_VERSION,
                        "attempt_id": attempt_id,
                        "plan_hash": plan["plan_hash"],
                        "contextual_preflight": preflight_card,
                        "gray_zone_model_call_count": 0,
                    },
                }
                batch.status = "extracting_graph"
                batch.completed_at = None
                batch.last_error = None
                batch.worker_id = current_worker_id()
                batch.heartbeat_at = datetime.utcnow()
                batch.stats = {
                    **dict(batch.stats or {}),
                    "phase": "context_graph",
                    "parse_committed": True,
                    "versioned_graph_retry_execution_mode": execution_mode,
                    "active_versioned_graph_retry": {
                        "protocol_version": VERSIONED_GRAPH_RETRY_PROTOCOL_VERSION,
                        "attempt_id": attempt_id,
                        "plan_hash": plan["plan_hash"],
                    },
                    "ingestion_resource_lock": resource_lock.diagnostics(),
                }
                db.commit()
                lifecycle_started = True
                emit_ingestion_log(
                    batch_id,
                    "versioned_graph_retry_started",
                    "Resuming the exact compensated versioned graph transaction",
                    attempt_id=attempt_id,
                    plan_hash=plan["plan_hash"],
                    graph_failure_audit_hash=plan["graph_failure_audit_hash"],
                )

                context_state = await rebuild_context_graph(
                    db,
                    str(plan["knowledge_base_id"]),
                    batch_id=batch_id,
                    chunk_version_incremented=True,
                    contextual_index_repair_mode=(
                        CONTEXTUAL_INDEX_REPAIR_MODE_VERIFY_ONLY
                    ),
                )
                graph_write_set = _graph_committed_write_set(context_state)
                batch = db.get(IngestionBatch, batch_id, with_for_update=True)
                recovery = db.get(
                    IngestionBatchRecovery,
                    str(plan["recovery_id"]),
                    with_for_update=True,
                )
                if batch is None or recovery is None:
                    raise RuntimeError(
                        "Versioned graph retry target disappeared before commit"
                    )
                completion_history = list(
                    (recovery.diagnostics_json or {}).get(
                        "versioned_graph_retry_history"
                    )
                    or []
                )
                completion_history.append(
                    {
                        "attempt_id": attempt_id,
                        "plan_hash": plan["plan_hash"],
                        "status": "completed",
                        "context_graph_state_id": str(context_state.id),
                        "completed_at": datetime.utcnow().isoformat(),
                    }
                )
                recovery.graph_write_set_json = graph_write_set
                recovery.graph_write_set_hash = _canonical_payload_hash(
                    graph_write_set
                )
                recovery.status = "completed"
                recovery.completed_at = datetime.utcnow()
                recovery.diagnostics_json = {
                    **dict(recovery.diagnostics_json or {}),
                    "active_versioned_graph_retry": None,
                    "versioned_graph_retry_history": completion_history,
                }
                recovery.compensation_json = {
                    **dict(recovery.compensation_json or {}),
                    "cache_invalidation": (
                        _versioned_graph_commit_cache_invalidation_state(
                            knowledge_base_id=str(recovery.knowledge_base_id),
                            attempt_id=attempt_id,
                            plan_hash=str(plan["plan_hash"]),
                            context_graph_state_id=str(context_state.id),
                            graph_write_set_hash=str(
                                recovery.graph_write_set_hash
                            ),
                        )
                    ),
                }
                batch.status = "completed"
                batch.completed_at = datetime.utcnow()
                batch.worker_id = None
                batch.heartbeat_at = None
                batch.last_error = None
                batch.stats = _completed_versioned_graph_retry_stats(
                    dict(batch.stats or {}),
                    context_state=context_state,
                    graph_write_set_hash=str(recovery.graph_write_set_hash),
                    attempt_id=attempt_id,
                    plan_hash=str(plan["plan_hash"]),
                )
                db.commit()
                graph_commit_completed = True
                lifecycle_started = False
                cache_state = _dispatch_batch_recovery_cache_invalidation(
                    db,
                    recovery_id=str(recovery.id),
                )
                batch = db.get(IngestionBatch, batch_id)
                if batch is None:
                    raise RuntimeError(
                        "Committed versioned graph retry batch disappeared"
                    )
                batch.stats = {
                    **dict(batch.stats or {}),
                    "post_commit_cache_invalidation": cache_state,
                }
                db.commit()
                emit_ingestion_log(
                    batch_id,
                    "versioned_graph_retry_completed",
                    "Exact versioned TPE and Four-Layer graph transaction completed",
                    attempt_id=attempt_id,
                    plan_hash=plan["plan_hash"],
                    context_graph_state_id=str(context_state.id),
                    cache_invalidation_status=cache_state.get("status"),
                )
                return summarize_batch(batch)
            except (Exception, asyncio.CancelledError) as exc:
                db.rollback()
                if graph_commit_completed:
                    try:
                        emit_ingestion_log(
                            batch_id,
                            "versioned_graph_retry_post_commit_followup_failed",
                            "Versioned graph retry committed; a post-commit follow-up failed",
                            plan_hash=(plan or {}).get("plan_hash"),
                            failure_type=type(exc).__name__[:128],
                            graph_commit_preserved=True,
                        )
                    except Exception:
                        pass
                    raise
                if not lifecycle_started:
                    raise
                safe_message = public_exception_message(
                    exc,
                    fallback="Versioned graph retry failed",
                )
                failure_audit = graph_build_failure_audit(exc)
                recovery = db.get(
                    IngestionBatchRecovery,
                    str(plan["recovery_id"]),
                )
                compensation: dict[str, Any] = {}
                compensation_error: str | None = None
                if (
                    recovery is not None
                    and recovery.parse_committed
                    and recovery.status == "graph_building"
                ):
                    try:
                        compensation = _restore_graph_before_scope(
                            db,
                            recovery=recovery,
                            reason=(
                                "versioned graph retry failed: " + safe_message
                            ),
                        )
                    except Exception as compensation_exc:
                        db.rollback()
                        compensation_error = public_exception_message(
                            compensation_exc,
                            fallback="Versioned graph retry compensation failed",
                        )
                batch = db.get(IngestionBatch, batch_id)
                recovery = db.get(
                    IngestionBatchRecovery,
                    str(plan["recovery_id"]),
                )
                if batch is not None:
                    batch.status = (
                        "cancel_failed" if compensation_error else "failed"
                    )
                    batch.completed_at = datetime.utcnow()
                    batch.worker_id = None
                    batch.heartbeat_at = None
                    batch.last_error = compensation_error or safe_message
                    batch.stats = {
                        **dict(batch.stats or {}),
                        "phase": "versioned_graph_retry_failed",
                        "active_versioned_graph_retry": None,
                        "versioned_graph_retry_failure": {
                            "protocol_version": (
                                VERSIONED_GRAPH_RETRY_PROTOCOL_VERSION
                            ),
                            "plan_hash": plan["plan_hash"],
                            "failure_audit": failure_audit,
                            "compensation": compensation,
                            "compensation_error": compensation_error,
                        },
                        "manual_review_required": bool(compensation_error),
                    }
                if recovery is not None:
                    failure_history = list(
                        (recovery.diagnostics_json or {}).get(
                            "versioned_graph_retry_history"
                        )
                        or []
                    )
                    failure_history.append(
                        {
                            "plan_hash": plan["plan_hash"],
                            "status": "failed",
                            "failure_type": type(exc).__name__[:128],
                            "failed_at": datetime.utcnow().isoformat(),
                        }
                    )
                    recovery.diagnostics_json = {
                        **dict(recovery.diagnostics_json or {}),
                        "active_versioned_graph_retry": None,
                        "versioned_graph_retry_history": failure_history,
                    }
                db.commit()
                emit_ingestion_log(
                    batch_id,
                    "versioned_graph_retry_failed",
                    "Versioned graph retry failed after exact compensation",
                    plan_hash=plan["plan_hash"],
                    failure_type=type(exc).__name__[:128],
                    manual_review_required=bool(compensation_error),
                )
                raise
            finally:
                db.close()
    finally:
        outer.close()


def plan_versioned_graph_completion_reconcile(
    db: Session,
    *,
    batch_id: str,
) -> dict[str, Any]:
    batch = db.get(IngestionBatch, batch_id)
    if batch is None or batch.status != "completed":
        raise RuntimeError(
            "Versioned graph completion reconcile requires a completed batch"
        )
    recovery = db.scalar(
        select(IngestionBatchRecovery).where(
            IngestionBatchRecovery.batch_id == batch_id
        )
    )
    if recovery is None or recovery.status != "completed":
        raise RuntimeError(
            "Versioned graph completion reconcile requires a completed recovery"
        )
    stats = dict(batch.stats or {})
    completion = dict(stats.get("versioned_graph_retry_completed") or {})
    if completion.get("protocol_version") != VERSIONED_GRAPH_RETRY_PROTOCOL_VERSION:
        raise RuntimeError("Versioned graph completion receipt is missing or invalid")
    attempt_id = str(completion.get("attempt_id") or "")
    plan_hash = str(completion.get("plan_hash") or "")
    context_state_id = str(completion.get("context_graph_state_id") or "")
    if (
        not attempt_id
        or len(plan_hash) != 64
        or not context_state_id
        or completion.get("gray_zone_model_call_count") != 0
    ):
        raise RuntimeError("Versioned graph completion receipt is incomplete")
    graph_write_set = dict(recovery.graph_write_set_json or {})
    graph_write_set_hash = _canonical_payload_hash(graph_write_set)
    if (
        not recovery.graph_write_set_hash
        or graph_write_set_hash != recovery.graph_write_set_hash
        or str(graph_write_set.get("context_graph_state_id") or "")
        != context_state_id
    ):
        raise RuntimeError("Versioned graph completion write-set identity drifted")
    context_state = db.get(ContextGraphState, context_state_id)
    if (
        context_state is None
        or context_state.knowledge_base_id != batch.knowledge_base_id
        or context_state.state != "active"
    ):
        raise RuntimeError("Versioned graph completion context state is not active")
    layer_ids = {
        "context": str(context_state.id),
        "relation": str(context_state.chunk_relation_graph_state_id or ""),
        "mid": str(context_state.mid_concept_state_id or ""),
        "coarse": str(context_state.coarse_concept_state_id or ""),
    }
    write_set_layer_ids = {
        "context": str(graph_write_set.get("context_graph_state_id") or ""),
        "relation": str(
            graph_write_set.get("chunk_relation_graph_state_id") or ""
        ),
        "mid": str(graph_write_set.get("mid_concept_state_id") or ""),
        "coarse": str(graph_write_set.get("coarse_concept_state_id") or ""),
    }
    if layer_ids != write_set_layer_ids or any(not value for value in layer_ids.values()):
        raise RuntimeError("Versioned graph completion layer identity drifted")
    active_layer_ids = {
        "context": sorted(
            str(value)
            for value in db.scalars(
                select(ContextGraphState.id).where(
                    ContextGraphState.knowledge_base_id == batch.knowledge_base_id,
                    ContextGraphState.state == "active",
                )
            ).all()
        ),
        "relation": sorted(
            str(value)
            for value in db.scalars(
                select(ChunkRelationGraphState.id).where(
                    ChunkRelationGraphState.knowledge_base_id
                    == batch.knowledge_base_id,
                    ChunkRelationGraphState.state == "active",
                )
            ).all()
        ),
        "mid": sorted(
            str(value)
            for value in db.scalars(
                select(MidConceptState.id).where(
                    MidConceptState.knowledge_base_id == batch.knowledge_base_id,
                    MidConceptState.state == "active",
                )
            ).all()
        ),
        "coarse": sorted(
            str(value)
            for value in db.scalars(
                select(CoarseConceptState.id).where(
                    CoarseConceptState.knowledge_base_id
                    == batch.knowledge_base_id,
                    CoarseConceptState.state == "active",
                )
            ).all()
        ),
    }
    if any(active_layer_ids[key] != [layer_ids[key]] for key in layer_ids):
        raise RuntimeError("Versioned graph completion active layer scope drifted")
    pointer = db.scalar(
        select(KnowledgeBaseVectorRuntimeState).where(
            KnowledgeBaseVectorRuntimeState.knowledge_base_id
            == batch.knowledge_base_id
        )
    )
    if pointer is None or {
        "context": str(pointer.active_context_graph_state_id or ""),
        "relation": str(pointer.active_chunk_relation_graph_state_id or ""),
        "mid": str(pointer.active_mid_concept_state_id or ""),
        "coarse": str(pointer.active_coarse_concept_state_id or ""),
    } != layer_ids:
        raise RuntimeError("Versioned graph completion vector pointer drifted")
    expected_cache_state = _versioned_graph_commit_cache_invalidation_state(
        knowledge_base_id=str(batch.knowledge_base_id),
        attempt_id=attempt_id,
        plan_hash=plan_hash,
        context_graph_state_id=context_state_id,
        graph_write_set_hash=graph_write_set_hash,
    )
    cache_state = dict(
        (recovery.compensation_json or {}).get("cache_invalidation") or {}
    )
    cache_binding_matches = bool(
        cache_state.get("graph_commit_binding_hash")
        == expected_cache_state["graph_commit_binding_hash"]
    )
    summary_reconcile_required = any(
        key in stats
        for key in (
            "context_graph_phase",
            "context_graph_heartbeat_at",
            "context_graph_metrics",
            "errors",
            "versioned_graph_retry_failure",
            "failure_compensation",
            "failed_at",
        )
    )
    cache_reconcile_required = bool(
        not cache_binding_matches or cache_state.get("status") != "dispatched"
    )
    card = {
        "protocol_version": "versioned_graph_completion_reconcile_v1",
        "batch_id": str(batch.id),
        "recovery_id": str(recovery.id),
        "knowledge_base_id": str(batch.knowledge_base_id),
        "attempt_id": attempt_id,
        "versioned_graph_retry_plan_hash": plan_hash,
        "context_graph_state_id": context_state_id,
        "layer_ids": layer_ids,
        "active_layer_ids": active_layer_ids,
        "graph_write_set_hash": graph_write_set_hash,
        "cache_binding_hash": expected_cache_state[
            "graph_commit_binding_hash"
        ],
        "cache_binding_matches": cache_binding_matches,
        "cache_status": cache_state.get("status"),
        "summary_reconcile_required": summary_reconcile_required,
        "cache_reconcile_required": cache_reconcile_required,
        "provider_call_count": 0,
        "qdrant_write_count": 0,
        "gray_zone_model_call_count": 0,
    }
    card["reconcile_required"] = bool(
        summary_reconcile_required or cache_reconcile_required
    )
    card["plan_hash"] = _canonical_payload_hash(card)
    return card


async def reconcile_versioned_graph_completion(
    *,
    batch_id: str,
    expected_plan_hash: str,
) -> dict[str, Any]:
    from app.db import SessionLocal

    admission = SessionLocal()
    try:
        admitted = plan_versioned_graph_completion_reconcile(
            admission,
            batch_id=batch_id,
        )
        if str(admitted["plan_hash"]) != str(expected_plan_hash):
            raise RuntimeError("Versioned graph completion reconcile plan is stale")
    finally:
        admission.rollback()
        admission.close()
    outer = SessionLocal()
    try:
        async with knowledge_base_ingestion_resource_lock(
            outer,
            str(admitted["knowledge_base_id"]),
            operation="versioned_graph_completion_reconcile",
            batch_id=batch_id,
        ):
            db = SessionLocal()
            try:
                plan = plan_versioned_graph_completion_reconcile(
                    db,
                    batch_id=batch_id,
                )
                if plan != admitted:
                    raise RuntimeError(
                        "Versioned graph completion reconcile plan drifted"
                    )
                if not bool(plan["reconcile_required"]):
                    batch = db.get(IngestionBatch, batch_id)
                    recovery = db.get(
                        IngestionBatchRecovery,
                        str(plan["recovery_id"]),
                    )
                    return {
                        "plan": plan,
                        "cache_invalidation": dict(
                            (recovery.compensation_json or {}).get(
                                "cache_invalidation"
                            )
                            or {}
                        )
                        if recovery is not None
                        else {},
                        "batch": summarize_batch(batch),
                        "idempotent_noop": True,
                    }
                batch = db.get(IngestionBatch, batch_id, with_for_update=True)
                recovery = db.get(
                    IngestionBatchRecovery,
                    str(plan["recovery_id"]),
                    with_for_update=True,
                )
                context_state = db.get(
                    ContextGraphState,
                    str(plan["context_graph_state_id"]),
                )
                if batch is None or recovery is None or context_state is None:
                    raise RuntimeError(
                        "Versioned graph completion reconcile target disappeared"
                    )
                completion = dict(
                    (batch.stats or {}).get("versioned_graph_retry_completed")
                    or {}
                )
                batch.stats = _completed_versioned_graph_retry_stats(
                    dict(batch.stats or {}),
                    context_state=context_state,
                    graph_write_set_hash=str(recovery.graph_write_set_hash),
                    attempt_id=str(completion["attempt_id"]),
                    plan_hash=str(completion["plan_hash"]),
                )
                expected_cache = _versioned_graph_commit_cache_invalidation_state(
                    knowledge_base_id=str(batch.knowledge_base_id),
                    attempt_id=str(completion["attempt_id"]),
                    plan_hash=str(completion["plan_hash"]),
                    context_graph_state_id=str(context_state.id),
                    graph_write_set_hash=str(recovery.graph_write_set_hash),
                )
                current_cache = dict(
                    (recovery.compensation_json or {}).get(
                        "cache_invalidation"
                    )
                    or {}
                )
                if (
                    current_cache.get("graph_commit_binding_hash")
                    != expected_cache["graph_commit_binding_hash"]
                ):
                    recovery.compensation_json = {
                        **dict(recovery.compensation_json or {}),
                        "cache_invalidation": expected_cache,
                    }
                db.commit()
                cache_state = _dispatch_batch_recovery_cache_invalidation(
                    db,
                    recovery_id=str(recovery.id),
                )
                batch = db.get(IngestionBatch, batch_id, with_for_update=True)
                if batch is None:
                    raise RuntimeError(
                        "Completed batch disappeared after cache invalidation"
                    )
                batch.stats = {
                    **dict(batch.stats or {}),
                    "post_commit_cache_invalidation": cache_state,
                    "versioned_graph_completion_reconcile": {
                        "protocol_version": (
                            "versioned_graph_completion_reconcile_v1"
                        ),
                        "plan_hash": str(plan["plan_hash"]),
                        "cache_binding_hash": str(plan["cache_binding_hash"]),
                        "cache_status": cache_state.get("status"),
                        "reconciled_at": datetime.utcnow().isoformat(),
                        "provider_call_count": 0,
                        "qdrant_write_count": 0,
                        "gray_zone_model_call_count": 0,
                    },
                }
                db.commit()
                return {
                    "plan": plan,
                    "cache_invalidation": cache_state,
                    "batch": summarize_batch(batch),
                }
            finally:
                db.close()
    finally:
        outer.close()


async def run_uploaded_files_ingestion(
    batch_id: str,
    file_paths: list[str],
    *,
    force: bool = False,
    full_reparse: bool = False,
    execution_mode: str = "inline",
) -> dict:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        if batch.status in TERMINAL_STATES:
            return summarize_batch(batch)
        _mark_batch_waiting_for_resource_lock(
            db,
            batch,
            operation="uploaded_files_ingestion",
        )
        async with knowledge_base_ingestion_resource_lock(
            db,
            batch.knowledge_base_id,
            operation="uploaded_files_ingestion",
            batch_id=batch.id,
        ):
            return await _run_uploaded_files_ingestion_locked(
                batch_id,
                file_paths,
                force=force,
                full_reparse=full_reparse,
                execution_mode=execution_mode,
            )
    except IngestionResourceBusyError as exc:
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            _record_batch_resource_lock_conflict(db, batch, exc)
        raise
    finally:
        db.close()


async def _run_uploaded_files_ingestion_locked(
    batch_id: str,
    file_paths: list[str],
    *,
    force: bool = False,
    full_reparse: bool = False,
    execution_mode: str = "inline",
) -> dict:
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} not found")
        if batch.status in TERMINAL_STATES:
            return summarize_batch(batch)
        ensure_not_cancelled(db, batch_id)
        knowledge_base = resolve_knowledge_base(db, batch.knowledge_base_id)
        resource_lock = active_ingestion_resource_lease(knowledge_base.id)
        if resource_lock is None:
            raise RuntimeError("Uploaded-files ingestion entered its versioning phase without the knowledge-base resource lock")
        # Upload storage can be durably provisioned before the first parse
        # while the immutable attempt-snapshot root does not exist yet.  Build
        # both per-KB roots only after the KB resource lock is held so a
        # raw-only or repaired KB cannot fail every file before parsing merely
        # because its ingestion root has not been created.
        ensure_knowledge_base_storage_durability_ready(
            knowledge_base.name,
            knowledge_base_source_root=knowledge_base.source_root,
            create_missing=True,
        )
        paths = [Path(path).resolve() for path in file_paths]
        if len({str(path) for path in paths}) != len(paths):
            raise RuntimeError("An ingestion batch cannot contain the same source path more than once")
        current_version = knowledge_base.current_chunk_version or 0
        active_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
        active_max_version = db.scalar(select(func.max(Chunk.chunk_version)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
        if full_reparse and active_chunks <= 0:
            raise RuntimeError("Full reparse is unavailable before the first successful parse")
        target_version = target_chunk_version(current_version=current_version, active_max_version=active_max_version, full_reparse=full_reparse)
        recovery = _prepare_batch_recovery(
            db,
            batch=batch,
            knowledge_base=knowledge_base,
            target_version=target_version,
            full_reparse=full_reparse,
        )
        batch = db.get(IngestionBatch, batch_id)
        knowledge_base = db.get(KnowledgeBase, recovery.knowledge_base_id)
        if batch is None or knowledge_base is None:
            raise RuntimeError("Ingestion batch recovery scope disappeared before parsing")
        batch.status = "parsing"
        batch.started_at = datetime.utcnow()
        batch.completed_at = None
        batch.total_files = len(paths)
        batch.processed_files = 0
        batch.success_count = 0
        batch.failure_count = 0
        batch.skipped_count = 0
        batch.worker_id = current_worker_id()
        batch.heartbeat_at = datetime.utcnow()
        batch.stats = {
            **(batch.stats or {}),
            "phase": "parsing",
            "target_version": target_version,
            "current_chunk_version_before": current_version,
            "active_max_chunk_version_before": active_max_version,
            "v_before_batch": recovery.v_before_batch,
            "batch_recovery_id": recovery.id,
            "batch_recovery_protocol_version": recovery.protocol_version,
            "full_reparse": full_reparse,
            "force": force,
            "ingestion_execution_mode": execution_mode,
            "ingestion_resource_lock": resource_lock.diagnostics(),
            "errors": [],
        }
        db.commit()
        emit_ingestion_log(batch.id, "batch_started", f"Parsing {len(paths)} files into context graph", total_files=len(paths), full_reparse=full_reparse)
        coverage: Counter[str] = Counter()
        language_coverage: Counter[str] = Counter()
        errors: list[dict[str, str]] = []
        for index, path in enumerate(paths, start=1):
            ensure_not_cancelled(db, batch_id)
            batch = db.get(IngestionBatch, batch_id)
            if batch is None:
                break
            recovery = db.get(IngestionBatchRecovery, recovery.id)
            if recovery is None:
                raise RuntimeError("Ingestion batch recovery row disappeared during parsing")
            file_stage = _prepare_file_stage(
                db,
                recovery=recovery,
                source_path=path,
                sequence_index=index,
            )
            batch = db.get(IngestionBatch, batch_id)
            if batch is None:
                raise RuntimeError(f"Batch {batch_id} disappeared before file parsing")
            mark_batch_worker_heartbeat(db, batch, phase="parsing")
            db.commit()
            emit_ingestion_log(batch.id, "file_started", f"[{index}/{len(paths)}] Parsing {path.name}", source_path=str(path))
            try:
                result = await ingest_file(
                    db,
                    path,
                    trigger_source=batch.trigger_source,
                    batch_id=batch.id,
                    knowledge_base_id=knowledge_base.id,
                    rebuild_graph=False,
                    force=force,
                    target_version=target_version,
                    file_stage_id=file_stage.id,
                )
                coverage[result["source_type"]] += 1
                language_coverage[str(result.get("language") or "unknown")] += 1
                batch.success_count += 1
                emit_ingestion_log(batch.id, "file_completed", f"{path.name} parsed successfully", source_path=str(path), stats=result.get("stats", {}))
            except IngestionCancelled as exc:
                db.rollback()
                _mark_file_stage_failed(
                    db,
                    stage_id=file_stage.id,
                    exc=exc,
                    cancelled=True,
                )
                raise
            except Exception as exc:
                db.rollback()
                message = exception_message(exc)
                _mark_file_stage_failed(
                    db,
                    stage_id=file_stage.id,
                    exc=exc,
                    cancelled=False,
                )
                errors.append({"source_path": str(path), "message": message})
                batch = db.get(IngestionBatch, batch_id)
                if batch is not None:
                    batch.failure_count += 1
                    batch.last_error = message
                    batch.stats = {**(batch.stats or {}), "errors": errors}
                emit_ingestion_log(batch_id, "file_failed", f"{path.name} parse failed: {message}", source_path=str(path), error=message)
                db.commit()
            finally:
                batch = db.get(IngestionBatch, batch_id)
                if batch is not None:
                    batch.processed_files += 1
                    batch.stats = {
                        **(batch.stats or {}),
                        "coverage_by_source_type": dict(coverage),
                        "coverage_by_language": dict(language_coverage),
                        "errors": errors,
                    }
                    db.commit()
                    emit_ingestion_log(
                        batch.id,
                        "batch_progress",
                        f"Progress {batch.processed_files}/{batch.total_files}",
                        processed_files=batch.processed_files,
                        total_files=batch.total_files,
                        success_count=batch.success_count,
                        failure_count=batch.failure_count,
                    )
        batch = db.get(IngestionBatch, batch_id)
        if batch is None:
            raise RuntimeError(f"Batch {batch_id} disappeared")
        ensure_not_cancelled(db, batch_id)
        if batch.success_count > 0:
            recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
            if recovery is None:
                raise RuntimeError("Ingestion batch recovery row disappeared at parse commit")
            _validate_batch_recovery(db, recovery)
            recovery.parse_committed = True
            recovery.status = "graph_building"
            recovery.diagnostics_json = {
                **dict(recovery.diagnostics_json or {}),
                "parse_committed_at": datetime.utcnow().isoformat(),
                "parse_commit_boundary": "before_context_graph_build",
                "parse_scope_rollback_after_boundary_allowed": False,
            }
            for file_stage in db.scalars(
                select(IngestionFileStage).where(
                    IngestionFileStage.batch_recovery_id == recovery.id,
                    IngestionFileStage.status == "indexed_committed",
                )
            ).all():
                _validated_file_stage(
                    db,
                    file_stage,
                    recovery=recovery,
                    require_write_set=True,
                )
                file_stage.status = "retained_after_parse_commit"
                file_stage.phase = "context_graph"
            batch.status = "extracting_graph"
            batch.stats = {
                **(batch.stats or {}),
                "phase": "context_graph",
                "parse_committed": True,
                "parse_commit_boundary": "before_context_graph_build",
            }
            mark_batch_worker_heartbeat(db, batch, phase="context_graph")
            db.commit()
            emit_ingestion_log(batch.id, "context_graph_started", "Building four-layer context graph")
            context_state = await rebuild_context_graph(db, knowledge_base.id, batch_id=batch.id, chunk_version_incremented=target_version > current_version)
            knowledge_base.current_chunk_version = max(knowledge_base.current_chunk_version or 0, target_version)
            graph_stats = dict(context_state.stats_json or {})
            graph_write_set = _graph_committed_write_set(context_state)
            recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
            if recovery is None or not recovery.parse_committed or recovery.status != "graph_building":
                raise RuntimeError("Ingestion graph completion lost its durable parse boundary")
            recovery.graph_write_set_json = graph_write_set
            recovery.graph_write_set_hash = _canonical_payload_hash(graph_write_set)
            recovery.status = "completed"
            recovery.completed_at = datetime.utcnow()
            batch.stats = {
                **(batch.stats or {}),
                "graph_stats": graph_stats,
                "phase": "completed",
                "parse_committed": True,
                "graph_write_set_hash": recovery.graph_write_set_hash,
            }
            emit_ingestion_log(batch.id, "context_graph_completed", "Four-layer context graph is active", **graph_stats)
        else:
            graph_stats = {}
            recovery = db.get(IngestionBatchRecovery, recovery.id, with_for_update=True)
            if recovery is None:
                raise RuntimeError("Ingestion recovery row disappeared before no-write completion")
            recovery.status = "completed_no_writes"
            recovery.completed_at = datetime.utcnow()
            batch.stats = {**(batch.stats or {}), "graph_stats": graph_stats, "phase": "completed", "parse_committed": False}
        if batch.success_count == 0 and batch.total_files > 0:
            batch.status = "failed"
            terminal_event = "batch_failed"
        elif batch.failure_count > 0:
            batch.status = "partial_failed"
            terminal_event = "batch_partial_failed"
        else:
            batch.status = "completed"
            terminal_event = "batch_completed"
        batch.completed_at = datetime.utcnow()
        batch.worker_id = None
        batch.heartbeat_at = None
        db.commit()
        emit_ingestion_log(batch.id, terminal_event, f"Batch {batch.status}: success={batch.success_count}, failed={batch.failure_count}, skipped={batch.skipped_count}")
        return summarize_batch(batch)
    except IngestionCancelled:
        db.rollback()
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            recovery = db.scalar(
                select(IngestionBatchRecovery).where(
                    IngestionBatchRecovery.batch_id == batch_id
                )
            )
            try:
                if recovery is not None and recovery.parse_committed:
                    compensation = _restore_graph_before_scope(
                        db,
                        recovery=recovery,
                        reason="cooperative cancellation during context graph build",
                    )
                    cancellation_status = "graph_cancelled_parse_scope_preserved"
                elif recovery is not None:
                    compensation = compensate_ingestion_batch_parse_writes(
                        db,
                        recovery=recovery,
                        reason="cooperative cancellation before parse commit boundary",
                    )
                    cancellation_status = "parse_writes_compensated"
                else:
                    compensation = {}
                    cancellation_status = "cancelled_by_worker"
            except Exception as compensation_exc:
                db.rollback()
                batch = db.get(IngestionBatch, batch_id)
                if batch is not None:
                    batch.status = "cancel_failed"
                    batch.last_error = exception_message(compensation_exc)
                    batch.completed_at = datetime.utcnow()
                    batch.worker_id = None
                    batch.heartbeat_at = None
                    batch.stats = {
                        **(batch.stats or {}),
                        "phase": "cancel_failed",
                        "cancel_requested": True,
                        "cancellation_status": "durable_compensation_pending_retry",
                        "manual_review_required": True,
                        "cancel_failure_reason": batch.last_error,
                    }
                    db.commit()
                raise
            batch = db.get(IngestionBatch, batch_id)
            if batch is None:
                raise RuntimeError("Batch disappeared after cancellation compensation")
            batch.stats = {
                **(batch.stats or {}),
                "cancellation_compensation": compensation,
            }
            mark_batch_cancelled(db, batch, cancellation_status=cancellation_status)
            db.commit()
            emit_ingestion_log(
                batch.id,
                "batch_cancelled",
                "Batch cancelled after durable compensation",
                state="cancelled",
                cancellation_status=cancellation_status,
            )
            return summarize_batch(batch)
        raise
    except Exception as exc:
        db.rollback()
        message = exception_message(exc)
        failure_audit = graph_build_failure_audit(exc)
        batch = db.get(IngestionBatch, batch_id)
        if batch is not None:
            recovery = db.scalar(
                select(IngestionBatchRecovery).where(
                    IngestionBatchRecovery.batch_id == batch_id
                )
            )
            recovery_result: dict[str, Any] | None = None
            recovery_error: str | None = None
            if recovery is not None and recovery.status not in {
                "completed",
                "completed_no_writes",
                "parse_compensated",
                "graph_compensated",
            }:
                try:
                    recovery_result = (
                        _restore_graph_before_scope(
                            db,
                            recovery=recovery,
                            reason=f"batch failure after parse commit: {message}",
                        )
                        if recovery.parse_committed
                        else compensate_ingestion_batch_parse_writes(
                            db,
                            recovery=recovery,
                            reason=f"batch failure before parse commit: {message}",
                        )
                    )
                except Exception as recovery_exc:
                    db.rollback()
                    recovery_error = exception_message(recovery_exc)
            batch = db.get(IngestionBatch, batch_id)
            if batch is None:
                raise RuntimeError("Batch disappeared during failure compensation")
            stats = dict(batch.stats or {})
            errors = list(stats.get("errors") or [])
            errors.append({"phase": stats.get("phase") or batch.status, "message": message})
            batch.status = "cancel_failed" if recovery_error else "failed"
            batch.last_error = recovery_error or message
            batch.completed_at = datetime.utcnow()
            batch.worker_id = None
            batch.heartbeat_at = None
            batch.stats = {
                **stats,
                "phase": "failed",
                "errors": errors,
                "manual_review_required": bool(recovery_error),
                "failure_compensation": recovery_result or {},
                "graph_failure_audit": failure_audit,
                **({"compensation_failure": recovery_error} if recovery_error else {}),
                "failed_at": datetime.utcnow().isoformat(),
            }
            db.commit()
            emit_ingestion_log(batch.id, "batch_failed", f"Batch failed during {(stats.get('phase') or batch.status)}: {message}", error=message)
        raise
    finally:
        db.close()


def target_chunk_version(*, current_version: int, active_max_version: int, full_reparse: bool) -> int:
    effective_current_version = max(current_version or 0, active_max_version or 0)
    return effective_current_version + 1 if full_reparse else (effective_current_version or 1)


async def run_ingestion_job(job_id: str, source_path: Path, trigger_source: str = "upload") -> dict:
    from app.db import SessionLocal

    with SessionLocal() as db:
        job = db.get(IngestionJob, job_id)
        knowledge_base_id = job.knowledge_base_id if job else None
        return await ingest_file(db, source_path, trigger_source=trigger_source, existing_job_id=job_id, knowledge_base_id=knowledge_base_id)


def list_knowledge_base_files(db: Session, knowledge_base_id: str) -> list[dict]:
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    if knowledge_base is None:
        return []
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"]
    storage_paths = {str(path) for path in collect_source_documents(storage_root)}
    documents = list(db.scalars(select(Document).where(Document.knowledge_base_id == knowledge_base_id, Document.is_active.is_(True))).all())
    documents_by_path = {document.source_path: document for document in documents}
    active_versions_by_document = {
        str(version.document_id): version
        for version in db.scalars(
            select(DocumentVersion).where(
                DocumentVersion.document_id.in_([document.id for document in documents]),
                DocumentVersion.is_active.is_(True),
            )
        ).all()
    }
    chunk_counts = Counter(
        {
            document_id: count
            for document_id, count in db.execute(
                select(Chunk.document_id, func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state == "active").group_by(Chunk.document_id)
            ).all()
        }
    )
    items: list[dict] = []
    for path_string in sorted(storage_paths | set(documents_by_path), key=str.lower):
        path = Path(path_string)
        document = documents_by_path.get(path_string)
        active_version = active_versions_by_document.get(str(document.id)) if document else None
        document_language_identity = language_identity_from_record(document)
        version_language_identity = language_identity_from_record(active_version)
        language_identity_consistent = bool(
            document_language_identity["valid"]
            and version_language_identity["valid"]
            and document_language_identity["detection_hash"]
            == version_language_identity["detection_hash"]
        )
        latest_job = db.scalar(select(IngestionJob).where(IngestionJob.knowledge_base_id == knowledge_base_id, IngestionJob.source_path == path_string).order_by(IngestionJob.updated_at.desc()))
        chunk_count = chunk_counts.get(document.id, 0) if document else 0
        status = "parsed" if chunk_count else "pending"
        if latest_job and is_active_file_job(latest_job):
            status = "parsing"
        elif latest_job and latest_job.status == "failed":
            status = "failed"
        items.append(
            {
                "id": document.id if document else path_string,
                "document_id": document.id if document else None,
                "title": document.title if document else path.stem,
                "source_path": path_string,
                "source_type": document.source_type if document else source_type_from_path(path),
                "partition": (document.tags or [None])[0] if document and document.tags else derive_partition(path, knowledge_base_name=knowledge_base.name),
                "language": (
                    version_language_identity["language"]
                    if language_identity_consistent
                    else None
                ),
                "language_source": (
                    version_language_identity["source"]
                    if language_identity_consistent
                    else None
                ),
                "language_confidence": (
                    version_language_identity["confidence"]
                    if language_identity_consistent
                    else None
                ),
                "language_detection_protocol_version": (
                    version_language_identity["protocol_version"]
                    if language_identity_consistent
                    else None
                ),
                "language_detection_hash": (
                    version_language_identity["detection_hash"]
                    if language_identity_consistent
                    else None
                ),
                "language_identity_consistent": language_identity_consistent,
                "status": status,
                "job_state": latest_job.status if latest_job else None,
                "batch_id": latest_job.batch_id if latest_job else None,
                "error": latest_job.error_message if latest_job and latest_job.status == "failed" else None,
                "chunk_count": chunk_count,
                "current_version": int(active_version.version) if active_version else 0,
                "active_chunks": chunk_count,
                "checksum": active_version.checksum if active_version else None,
                "chunk_version": int(active_version.version) if active_version else None,
                "updated_at": document.updated_at if document else latest_job.updated_at if latest_job else None,
                "last_ingested_at": active_version.created_at if active_version else None,
            }
        )
    return items


def source_file_delete_schema_hash() -> str:
    return _canonical_payload_hash(
        {
            "protocol_version": SOURCE_FILE_DELETE_PROTOCOL_VERSION,
            "namespace_durability_protocol_allowlist": sorted(
                SOURCE_FILE_DELETE_NAMESPACE_DURABILITY_PROTOCOLS
            ),
            "payload_fields": list(SOURCE_FILE_DELETE_PAYLOAD_FIELDS),
            "file_state_fields": list(SOURCE_FILE_DELETE_FILE_STATE_FIELDS),
            "source_file_card_fields": list(SOURCE_FILE_DELETE_SOURCE_FILE_CARD_FIELDS),
            "document_state_fields": list(SOURCE_FILE_DELETE_DOCUMENT_STATE_FIELDS),
            "database_commit_fields": list(SOURCE_FILE_DELETE_DATABASE_COMMIT_FIELDS),
            "history_fields": list(SOURCE_FILE_DELETE_HISTORY_FIELDS),
            "status_phase_map": {
                key: sorted(value) for key, value in SOURCE_FILE_DELETE_STATUS_PHASES.items()
            },
            "path_protocol": "absolute_lexical_contained_nofollow_leaf_v1",
            "checksum_protocol": "sha256_streaming_stable_stat_v1",
            "transaction_protocol": "intent_then_durable_unlink_then_scope_freshness_commit_v1",
            "cache_protocol": "post_database_commit_kb_invalidation_v1",
        }
    )


def _source_file_delete_payload_hash(payload: dict[str, Any]) -> str:
    return _canonical_payload_hash(
        {key: value for key, value in payload.items() if key != "intent_payload_hash"}
    )


def _refresh_source_file_delete_payload_hash(payload: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(payload)
    refreshed["intent_payload_hash"] = _source_file_delete_payload_hash(refreshed)
    return refreshed


def _source_file_delete_now() -> str:
    return datetime.utcnow().isoformat()


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _source_file_delete_target(
    knowledge_base: KnowledgeBase,
    source_path: str,
) -> Path:
    raw_target = Path(source_path)
    if not raw_target.is_absolute():
        raise UploadValidationError("Source deletion path must be absolute")
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )[
        "storage_root"
    ]
    target = contained_path(raw_target, storage_root)
    safe_name = normalize_upload_filename(target.name)
    if Path(safe_name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise UploadValidationError("Source deletion target has an unsupported file type")
    return target


def _source_file_delete_file_state(
    path: Path,
    authorized_root: Path,
) -> dict[str, Any]:
    state, _identity = _source_file_delete_verified_file_state(path, authorized_root)
    return state


def _source_file_delete_verified_file_state(
    path: Path,
    authorized_root: Path,
) -> tuple[dict[str, Any], VerifiedSourceIdentity | None]:
    if path.is_symlink():
        raise SourceFileDeleteIntegrityError("Source deletion target is a symbolic link")
    if not path.exists():
        return {"exists": False, "checksum": None, "size_bytes": 0}, None
    try:
        checksum, identity = verified_source_checksum(path, authorized_root)
    except Exception as exc:
        raise SourceFileDeleteIntegrityError(
            "Source deletion target failed final-open identity verification"
        ) from exc
    return (
        {
            "exists": True,
            "checksum": checksum,
            "size_bytes": int(identity.size_bytes),
        },
        identity,
    )


def _same_source_file_delete_state(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> bool:
    return (
        bool(actual.get("exists")) == bool(expected.get("exists"))
        and actual.get("checksum") == expected.get("checksum")
        and int(actual.get("size_bytes") or 0) == int(expected.get("size_bytes") or 0)
    )


def _source_file_delete_slot_key(
    db: Session,
    *,
    knowledge_base_id: str,
    source_path: str,
) -> str:
    document_keys = list(
        db.scalars(
            select(Document.logical_source_slot_key)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.source_path == source_path,
            )
            .order_by(Document.id.asc())
            .limit(2)
        ).all()
    )
    source_file_keys = list(
        db.scalars(
            select(SourceFile.logical_source_slot_key)
            .where(
                SourceFile.knowledge_base_id == knowledge_base_id,
                SourceFile.source_path == source_path,
            )
            .order_by(SourceFile.id.asc())
            .limit(2)
        ).all()
    )
    keys = {str(value) for value in [*document_keys, *source_file_keys] if value}
    if len(document_keys) > 1:
        raise SourceFileDeleteConflict(
            "Multiple documents claim the same physical source path"
        )
    if len(source_file_keys) > 1 or len(keys) > 1:
        raise SourceFileDeleteConflict(
            "Physical source path has ambiguous logical slot ownership"
        )
    if keys:
        return next(iter(keys))
    knowledge_base_source_root = db.scalar(
        select(KnowledgeBase.source_root).where(
            KnowledgeBase.id == knowledge_base_id
        )
    )
    if knowledge_base_source_root is None:
        raise SourceFileDeleteIntegrityError(
            "Knowledge base disappeared while resolving source slot identity"
        )
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base_source_root
    )["storage_root"]
    return source_slot_key_for_path(Path(source_path), storage_root)


def _source_file_delete_document_state(
    db: Session,
    *,
    knowledge_base_id: str,
    source_path: str,
    for_update: bool,
) -> tuple[Document | None, dict[str, Any], list[Chunk]]:
    logical_source_slot_key = _source_file_delete_slot_key(
        db,
        knowledge_base_id=knowledge_base_id,
        source_path=source_path,
    )
    query = (
        select(Document)
        .where(
            Document.knowledge_base_id == knowledge_base_id,
            Document.logical_source_slot_key == logical_source_slot_key,
        )
        .order_by(Document.id.asc())
        .limit(2)
        .execution_options(populate_existing=True)
    )
    if for_update:
        query = query.with_for_update()
    documents = list(db.scalars(query).all())
    if len(documents) > 1:
        raise SourceFileDeleteConflict(
            "Multiple documents claim the same source path; reconcile logical source identity before deletion"
        )
    document = documents[0] if documents else None
    if document is None:
        return (
            None,
            {
                "exists": False,
                "document_id": None,
                "source_path": None,
                "checksum": None,
                "is_active": False,
                "updated_at": None,
                "active_chunk_count": 0,
                "active_chunk_scope_hash": compute_chunk_scope_hash([]),
            },
            [],
        )
    chunk_query = (
        select(Chunk)
        .where(Chunk.document_id == document.id, Chunk.state == "active")
        .order_by(Chunk.chunk_index.asc(), Chunk.id.asc())
        .execution_options(populate_existing=True)
    )
    if for_update:
        chunk_query = chunk_query.with_for_update()
    active_chunks = list(db.scalars(chunk_query).all())
    return (
        document,
        {
            "exists": True,
            "document_id": document.id,
            "source_path": document.source_path,
            "checksum": document.checksum,
            "is_active": bool(document.is_active),
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "active_chunk_count": len(active_chunks),
            "active_chunk_scope_hash": compute_chunk_scope_hash(active_chunks),
        },
        active_chunks,
    )


def _source_file_delete_source_file_state(
    db: Session,
    *,
    knowledge_base_id: str,
    source_path: str,
    for_update: bool,
) -> tuple[list[SourceFile], list[dict[str, Any]]]:
    logical_source_slot_key = _source_file_delete_slot_key(
        db,
        knowledge_base_id=knowledge_base_id,
        source_path=source_path,
    )
    query = (
        select(SourceFile)
        .where(
            SourceFile.knowledge_base_id == knowledge_base_id,
            SourceFile.logical_source_slot_key == logical_source_slot_key,
            SourceFile.state == "active",
        )
        .order_by(SourceFile.id.asc())
        .limit(65)
        .execution_options(populate_existing=True)
    )
    if for_update:
        query = query.with_for_update()
    rows = list(db.scalars(query).all())
    if len(rows) > 64:
        raise SourceFileDeleteConflict(
            "More than 64 active source lifecycle rows claim one path; reconcile source identity before deletion"
        )
    cards = [
        {
            "source_file_id": str(row.id),
            "document_id": str(row.document_id) if row.document_id is not None else None,
            "source_path": str(row.source_path),
            "checksum": str(row.checksum),
            "source_type": str(row.source_type),
            "size_bytes": int(row.size_bytes or 0),
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
        for row in rows
    ]
    return rows, cards


def _empty_source_file_delete_database_commit() -> dict[str, Any]:
    return {
        "committed_at": None,
        "deleted_chunk_count": 0,
        "deleted_chunk_ids_sample": [],
        "deleted_chunk_ids_truncated_count": 0,
        "deleted_source_file_count": 0,
        "active_chunk_scope_hash_after": None,
        "knowledge_base_chunk_version_after": None,
        "chunk_version_state_id": None,
        "chunk_version_state_hash": None,
        "context_graph_state_id": None,
        "freshness_rows_marked_stale": 0,
        "cache_invalidation_required": True,
        "cache_invalidation_dispatched_at": None,
        "qdrant_cleanup_performed": False,
    }


def _validate_source_file_delete_payload(
    row: IngestionCompensationLog,
    knowledge_base: KnowledgeBase,
) -> tuple[dict[str, Any], Path]:
    payload = dict(row.payload_json or {})
    if set(payload) != set(SOURCE_FILE_DELETE_PAYLOAD_FIELDS):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an unexpected payload schema"
        )
    if payload.get("protocol_version") != SOURCE_FILE_DELETE_PROTOCOL_VERSION:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an unknown protocol"
        )
    if payload.get("namespace_durability_protocol") not in (
        SOURCE_FILE_DELETE_NAMESPACE_DURABILITY_PROTOCOLS
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an unsupported durability protocol"
        )
    if payload.get("schema_hash") != source_file_delete_schema_hash():
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an unknown schema hash"
        )
    if (
        payload.get("intent_id") != row.id
        or payload.get("knowledge_base_id") != row.knowledge_base_id
        or payload.get("knowledge_base_name") != knowledge_base.name
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} identity does not match its row"
        )
    allowed_phases = SOURCE_FILE_DELETE_STATUS_PHASES.get(str(row.status or ""))
    if allowed_phases is None or payload.get("phase") not in allowed_phases:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an invalid status/phase combination"
        )
    if not all(
        isinstance(payload.get(field), str) and payload[field]
        for field in ("created_at", "updated_at")
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has invalid timestamps"
        )
    if payload.get("intent_payload_hash") != _source_file_delete_payload_hash(payload):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} payload hash failed"
        )
    target = _source_file_delete_target(
        knowledge_base,
        str(payload.get("source_path") or ""),
    )
    if list(row.target_ids_json or []) != [str(target)]:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} target index does not match"
        )

    file_before = payload.get("file_before")
    if not isinstance(file_before, dict) or set(file_before) != set(
        SOURCE_FILE_DELETE_FILE_STATE_FIELDS
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has invalid file before-state"
        )
    if not isinstance(file_before.get("exists"), bool):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an invalid file existence flag"
        )
    if bool(file_before["exists"]):
        if (
            not _is_sha256_hex(file_before.get("checksum"))
            or not _is_nonnegative_int(file_before.get("size_bytes"))
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has invalid file identity"
            )
    elif file_before.get("checksum") is not None or file_before.get("size_bytes") != 0:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has invalid absent file identity"
        )

    document_before = payload.get("document_before")
    if not isinstance(document_before, dict) or set(document_before) != set(
        SOURCE_FILE_DELETE_DOCUMENT_STATE_FIELDS
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has invalid document before-state"
        )
    if not isinstance(document_before.get("exists"), bool):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an invalid document existence flag"
        )
    if bool(document_before["exists"]):
        if (
            not isinstance(document_before.get("document_id"), str)
            or not document_before["document_id"]
            or document_before.get("source_path") != str(target)
            or not _is_sha256_hex(document_before.get("checksum"))
            or not isinstance(document_before.get("is_active"), bool)
            or not isinstance(document_before.get("updated_at"), str)
            or not document_before["updated_at"]
            or not _is_nonnegative_int(document_before.get("active_chunk_count"))
            or not _is_sha256_hex(document_before.get("active_chunk_scope_hash"))
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has invalid document identity"
            )
    elif (
        document_before.get("document_id") is not None
        or document_before.get("source_path") is not None
        or document_before.get("checksum") is not None
        or document_before.get("is_active") is not False
        or document_before.get("updated_at") is not None
        or document_before.get("active_chunk_count") != 0
        or not _is_sha256_hex(document_before.get("active_chunk_scope_hash"))
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has invalid absent document identity"
        )
    if bool(file_before["exists"]) and bool(document_before["exists"]):
        if file_before["checksum"] != document_before["checksum"]:
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} file/document checksum fence failed"
            )

    source_files_before = payload.get("source_files_before")
    if not isinstance(source_files_before, list) or len(source_files_before) > 64:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has an invalid SourceFile before-state"
        )
    source_file_ids: set[str] = set()
    for card in source_files_before:
        if (
            not isinstance(card, dict)
            or set(card) != set(SOURCE_FILE_DELETE_SOURCE_FILE_CARD_FIELDS)
            or not isinstance(card.get("source_file_id"), str)
            or not card["source_file_id"]
            or card["source_file_id"] in source_file_ids
            or (
                card.get("document_id") is not None
                and (
                    not isinstance(card["document_id"], str)
                    or not card["document_id"]
                )
            )
            or card.get("source_path") != str(target)
            or not _is_sha256_hex(card.get("checksum"))
            or not isinstance(card.get("source_type"), str)
            or not card["source_type"]
            or not _is_nonnegative_int(card.get("size_bytes"))
            or not isinstance(card.get("updated_at"), str)
            or not card["updated_at"]
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has an invalid SourceFile identity card"
            )
        source_file_ids.add(card["source_file_id"])
    if (
        not file_before["exists"]
        and not document_before["exists"]
        and not source_files_before
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has no frozen target fact"
        )

    database_commit = payload.get("database_commit")
    if not isinstance(database_commit, dict) or set(database_commit) != set(
        SOURCE_FILE_DELETE_DATABASE_COMMIT_FIELDS
    ):
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has invalid database commit witness"
        )
    if database_commit.get("cache_invalidation_required") is not True:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} lost its cache invalidation requirement"
        )
    if database_commit.get("qdrant_cleanup_performed") is not False:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} falsely claims Qdrant cleanup"
        )
    database_committed = payload["phase"] in {"database_committed", "completed"}
    if database_committed:
        if (
            not isinstance(database_commit.get("committed_at"), str)
            or not database_commit["committed_at"]
            or not _is_nonnegative_int(database_commit.get("deleted_chunk_count"))
            or not isinstance(database_commit.get("deleted_chunk_ids_sample"), list)
            or len(database_commit["deleted_chunk_ids_sample"]) > 64
            or not _is_nonnegative_int(
                database_commit.get("deleted_chunk_ids_truncated_count")
            )
            or not _is_nonnegative_int(database_commit.get("deleted_source_file_count"))
            or not _is_sha256_hex(database_commit.get("active_chunk_scope_hash_after"))
            or not _is_nonnegative_int(
                database_commit.get("knowledge_base_chunk_version_after")
            )
            or not _is_nonnegative_int(
                database_commit.get("freshness_rows_marked_stale")
            )
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has an incomplete database commit witness"
            )
        state_id = database_commit.get("chunk_version_state_id")
        state_hash = database_commit.get("chunk_version_state_hash")
        if (state_id is None) != (state_hash is None) or (
            state_id is not None
            and (
                not isinstance(state_id, str)
                or not state_id
                or not _is_sha256_hex(state_hash)
                or database_commit["knowledge_base_chunk_version_after"] <= 0
            )
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has an invalid ChunkVersion witness"
            )
        if state_id is None and database_commit["knowledge_base_chunk_version_after"] != 0:
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} left an unproved current chunk version"
            )
        context_state_id = database_commit.get("context_graph_state_id")
        if context_state_id is not None and (
            not isinstance(context_state_id, str) or not context_state_id
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has an invalid context graph witness"
            )
        if context_state_id is None and database_commit["freshness_rows_marked_stale"] != 0:
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has freshness rows without a graph state"
            )
        if database_commit["deleted_source_file_count"] != len(source_files_before):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has an incomplete SourceFile witness"
            )
        deleted_ids = database_commit["deleted_chunk_ids_sample"]
        if (
            any(not isinstance(item, str) or not item for item in deleted_ids)
            or len(set(deleted_ids)) != len(deleted_ids)
            or len(deleted_ids) > database_commit["deleted_chunk_count"]
            or database_commit["deleted_chunk_ids_truncated_count"]
            != database_commit["deleted_chunk_count"] - len(deleted_ids)
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has an invalid deleted chunk witness"
            )
        if payload["phase"] == "database_committed" and database_commit.get(
            "cache_invalidation_dispatched_at"
        ) is not None:
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} claims cache dispatch before completion"
            )
        if payload["phase"] == "completed" and (
            not isinstance(database_commit.get("cache_invalidation_dispatched_at"), str)
            or not database_commit["cache_invalidation_dispatched_at"]
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} completed without cache invalidation"
            )
    elif database_commit != _empty_source_file_delete_database_commit():
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} recorded database facts before commit"
        )

    history = payload.get("history")
    if not isinstance(history, list) or len(history) > 32:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent {row.id} has invalid bounded history"
        )
    for event in history:
        if (
            not isinstance(event, dict)
            or not {"from", "to", "at"}.issubset(event)
            or not set(event).issubset(SOURCE_FILE_DELETE_HISTORY_FIELDS)
            or not isinstance(event.get("at"), str)
        ):
            raise SourceFileDeleteIntegrityError(
                f"Source deletion intent {row.id} has an invalid history event"
            )
    return payload, target


def _transition_source_file_delete_payload(
    payload: dict[str, Any],
    phase: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _source_file_delete_now()
    history = list(payload.get("history") or [])
    history.append(
        {
            "from": payload.get("phase"),
            "to": phase,
            "at": now,
            **({"details": details} if details else {}),
        }
    )
    return _refresh_source_file_delete_payload_hash(
        {
            **payload,
            "phase": phase,
            "updated_at": now,
            "history": history[-32:],
        }
    )


def pending_source_file_delete_intents(
    db: Session,
    knowledge_base_id: str,
) -> list[IngestionCompensationLog]:
    return list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(
                IngestionCompensationLog.knowledge_base_id == knowledge_base_id,
                IngestionCompensationLog.operation == SOURCE_FILE_DELETE_OPERATION,
                IngestionCompensationLog.status.in_(sorted(SOURCE_FILE_DELETE_ACTIVE_STATUSES)),
            )
            .order_by(
                IngestionCompensationLog.created_at.asc(),
                IngestionCompensationLog.id.asc(),
            )
            .limit(2)
            .execution_options(populate_existing=True)
        ).all()
    )


def source_file_delete_resource_fence_diagnostics(
    db: Session,
    *,
    knowledge_base_id: str,
    operation: str,
    owner_intent_id: str | None,
) -> dict[str, Any] | None:
    rows = pending_source_file_delete_intents(db, knowledge_base_id)
    if not rows:
        if operation == SOURCE_FILE_DELETE_LOCK_OPERATION and owner_intent_id:
            return {
                "reason": "source_file_delete_owner_missing",
                "retryable": True,
                "recovery_operation": SOURCE_FILE_DELETE_LOCK_OPERATION,
                "retry_guidance": "refresh the file list and retry the exact deletion",
            }
        return None
    if len(rows) > 1:
        return {
            "reason": "multiple_source_file_delete_intents",
            "retryable": False,
            "intent_ids": [row.id for row in rows],
            "retry_guidance": "run manual source-delete intent reconciliation",
        }
    row = rows[0]
    knowledge_base = db.get(
        KnowledgeBase,
        knowledge_base_id,
        populate_existing=True,
    )
    validation_error: str | None = None
    payload: dict[str, Any] = dict(row.payload_json or {})
    if knowledge_base is None:
        validation_error = "knowledge_base_missing"
    else:
        try:
            payload, _target = _validate_source_file_delete_payload(row, knowledge_base)
        except Exception as exc:
            validation_error = exc.__class__.__name__
    retryable = validation_error is None and row.status != "manual_review"
    if (
        retryable
        and operation == SOURCE_FILE_DELETE_LOCK_OPERATION
        and owner_intent_id == row.id
    ):
        return None
    return {
        "reason": "source_file_delete_recovery_pending",
        "intent_id": row.id,
        "status": row.status,
        "phase": payload.get("phase"),
        "protocol_version": payload.get("protocol_version"),
        "validation_error": validation_error,
        "retryable": retryable,
        "requested_owner_matches": owner_intent_id == row.id,
        "recovery_operation": SOURCE_FILE_DELETE_LOCK_OPERATION,
        "retry_guidance": (
            "retry the exact source-file deletion"
            if retryable
            else "run manual source-delete intent reconciliation"
        ),
    }


def source_file_delete_lock_token(
    db: Session,
    knowledge_base: KnowledgeBase,
    source_path: str,
) -> str | None:
    target = _source_file_delete_target(knowledge_base, source_path)
    rows = pending_source_file_delete_intents(db, knowledge_base.id)
    if not rows:
        return None
    if len(rows) > 1:
        raise SourceFileDeleteConflict(
            "Multiple durable source deletion intents require manual reconciliation"
        )
    row = rows[0]
    payload, pending_target = _validate_source_file_delete_payload(row, knowledge_base)
    if pending_target != target:
        raise SourceFileDeleteConflict(
            "Another source deletion is pending for this knowledge base; retry that exact deletion first"
        )
    if row.status == "manual_review":
        raise SourceFileDeleteConflict(
            "The pending source deletion requires manual reconciliation"
        )
    if payload.get("protocol_version") != SOURCE_FILE_DELETE_PROTOCOL_VERSION:
        raise SourceFileDeleteConflict("The pending source deletion protocol is unsupported")
    return row.id


def _require_source_file_delete_resource_lock(knowledge_base_id: str) -> Any:
    lease = active_ingestion_resource_lease(knowledge_base_id)
    if lease is None or lease.operation != SOURCE_FILE_DELETE_LOCK_OPERATION:
        raise SourceFileDeleteError(
            "Source deletion requires the knowledge-base ingestion resource lock"
        )
    return lease


def _load_source_file_delete_intent(
    db: Session,
    intent_id: str,
    *,
    for_update: bool = True,
) -> IngestionCompensationLog:
    query = select(IngestionCompensationLog).where(
        IngestionCompensationLog.id == intent_id,
        IngestionCompensationLog.operation == SOURCE_FILE_DELETE_OPERATION,
    ).execution_options(populate_existing=True)
    if for_update:
        query = query.with_for_update()
    row = db.scalar(query)
    if row is None:
        raise SourceFileDeleteIntegrityError(
            f"Source deletion intent does not exist: {intent_id}"
        )
    return row


def _prepare_source_file_delete_intent(
    db: Session,
    *,
    knowledge_base: KnowledgeBase,
    source_path: str,
) -> IngestionCompensationLog | None:
    from app.services.upload_replacement import pending_upload_replacement_blockers

    blockers = pending_upload_replacement_blockers(db, knowledge_base.id)
    if blockers:
        raise SourceFileDeleteConflict(
            "Cannot delete a source while durable upload replacement recovery is pending"
        )
    ensure_knowledge_base_storage_durability_ready(
        knowledge_base.name,
        knowledge_base_source_root=knowledge_base.source_root,
        create_missing=False,
    )
    durability_protocol = namespace_durability_protocol()
    if durability_protocol not in SOURCE_FILE_DELETE_NAMESPACE_DURABILITY_PROTOCOLS:
        raise SourceFileDeleteIntegrityError(
            "The current namespace durability protocol is unsupported for source deletion"
        )
    target = _source_file_delete_target(knowledge_base, source_path)
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"]
    file_before = _source_file_delete_file_state(target, storage_root)
    _document, document_before, _document_chunks = _source_file_delete_document_state(
        db,
        knowledge_base_id=knowledge_base.id,
        source_path=str(target),
        for_update=True,
    )
    _source_files, source_files_before = _source_file_delete_source_file_state(
        db,
        knowledge_base_id=knowledge_base.id,
        source_path=str(target),
        for_update=True,
    )
    if (
        not file_before["exists"]
        and not document_before["exists"]
        and not source_files_before
    ):
        return None
    if file_before["exists"] and document_before["exists"]:
        if file_before["checksum"] != document_before["checksum"]:
            raise SourceFileDeleteConflict(
                "Source bytes no longer match the document checksum; reconcile upload state before deletion"
            )
    intent_id = str(uuid4())
    now = _source_file_delete_now()
    payload = _refresh_source_file_delete_payload_hash(
        {
            "protocol_version": SOURCE_FILE_DELETE_PROTOCOL_VERSION,
            "namespace_durability_protocol": durability_protocol,
            "schema_hash": source_file_delete_schema_hash(),
            "intent_id": intent_id,
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "source_path": str(target),
            "phase": "intent_committed",
            "file_before": file_before,
            "document_before": document_before,
            "source_files_before": source_files_before,
            "database_commit": _empty_source_file_delete_database_commit(),
            "history": [],
            "created_at": now,
            "updated_at": now,
        }
    )
    row = IngestionCompensationLog(
        id=intent_id,
        knowledge_base_id=knowledge_base.id,
        operation=SOURCE_FILE_DELETE_OPERATION,
        target_ids_json=[str(target)],
        payload_json=payload,
        status="pending",
    )
    db.add(row)
    db.flush()
    return row


def _record_source_file_delete_failure(
    db: Session,
    *,
    intent_id: str,
    exc: BaseException,
) -> None:
    db.rollback()
    with db.begin():
        row = _load_source_file_delete_intent(db, intent_id)
        knowledge_base = db.get(
            KnowledgeBase,
            row.knowledge_base_id,
            populate_existing=True,
        )
        if knowledge_base is None:
            raise SourceFileDeleteRecoveryError(
                "Cannot preserve source deletion recovery because its knowledge base is missing"
            ) from exc
        try:
            payload, _target = _validate_source_file_delete_payload(row, knowledge_base)
        except SourceFileDeleteIntegrityError:
            row.status = "manual_review"
            row.error_message = exc.__class__.__name__
            return
        manual_review = isinstance(
            exc,
            (SourceFileDeleteIntegrityError, SourceFileDeleteConflict),
        )
        phase = str(payload["phase"])
        failed = _transition_source_file_delete_payload(
            payload,
            phase,
            details={"failure_type": exc.__class__.__name__},
        )
        row.payload_json = failed
        row.status = (
            "manual_review"
            if manual_review
            else (
                "cache_invalidation_pending"
                if phase == "database_committed"
                else "external_failed"
            )
        )
        row.error_message = exc.__class__.__name__


def _apply_source_file_delete_external_effect(
    *,
    knowledge_base_id: str,
    knowledge_base_name: str,
    knowledge_base_source_root: str,
    target: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    ensure_knowledge_base_storage_durability_ready(
        knowledge_base_name,
        knowledge_base_source_root=knowledge_base_source_root,
        create_missing=False,
    )
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base_source_root
    )["storage_root"]
    actual, verified_identity = _source_file_delete_verified_file_state(
        target,
        storage_root,
    )
    action: str
    if expected["exists"]:
        if not actual["exists"]:
            # A prior attempt may have completed unlink but failed while
            # publishing the directory entry removal.  Re-observation alone
            # is not a durability witness; exact retry must re-fsync parent.
            durable_sync_directory(target.parent)
            action = "already_absent_parent_sync_replayed"
        else:
            if not _same_source_file_delete_state(actual, expected):
                raise SourceFileDeleteIntegrityError(
                    "Source deletion target identity changed after intent commit"
                )
            if verified_identity is None:
                raise SourceFileDeleteIntegrityError(
                    "Source deletion target identity disappeared before durable unlink"
                )
            durable_unlink(
                target,
                missing_ok=False,
                expected_identity=verified_identity,
            )
            action = "durable_unlink_applied"
    else:
        if actual["exists"]:
            raise SourceFileDeleteIntegrityError(
                "An unplanned source file appeared after an absent-target intent"
            )
        action = "already_absent"
    if _source_file_delete_file_state(target, storage_root)["exists"]:
        raise SourceFileDeleteIntegrityError(
            "Source deletion postcondition failed: target still exists"
        )
    return {"action": action}


def _finalize_source_file_delete_database(
    db: Session,
    *,
    row: IngestionCompensationLog,
    knowledge_base: KnowledgeBase,
    payload: dict[str, Any],
    target: Path,
) -> None:
    document, current_document_state, document_chunks = _source_file_delete_document_state(
        db,
        knowledge_base_id=knowledge_base.id,
        source_path=str(target),
        for_update=True,
    )
    if current_document_state != payload["document_before"]:
        raise SourceFileDeleteIntegrityError(
            "Document or active chunk scope changed after source deletion intent commit"
        )

    source_files, current_source_file_state = _source_file_delete_source_file_state(
        db,
        knowledge_base_id=knowledge_base.id,
        source_path=str(target),
        for_update=True,
    )
    if current_source_file_state != payload["source_files_before"]:
        raise SourceFileDeleteIntegrityError(
            "SourceFile lifecycle changed after source deletion intent commit"
        )
    for source_file in source_files:
        source_file.state = "deleted"

    changed_chunk_ids: list[str] = []
    if document is not None:
        document.is_active = False
        for chunk in document_chunks:
            chunk.state = "deleted"
            changed_chunk_ids.append(str(chunk.id))
    db.flush()

    try:
        version_state = rewrite_chunk_version_state_for_active_scope_mutation(
            db,
            knowledge_base_id=knowledge_base.id,
            current_chunk_version=int(knowledge_base.current_chunk_version or 0),
        )
    except RuntimeError as exc:
        raise SourceFileDeleteIntegrityError(
            "ChunkVersion scope cannot be rewritten from the frozen deletion state"
        ) from exc
    active_scope_hash_after = (
        str((version_state.stats_json or {})["active_chunk_scope_hash"])
        if version_state is not None
        else compute_chunk_scope_hash([])
    )
    freshness = {
        "context_graph_state_id": None,
        "freshness_rows_marked_stale": 0,
    }
    if changed_chunk_ids:
        freshness = mark_context_graph_active_scope_stale(
            db,
            knowledge_base_id=knowledge_base.id,
            changed_chunk_ids=changed_chunk_ids,
            active_chunk_scope_hash=active_scope_hash_after,
            mutation=SOURCE_FILE_DELETE_OPERATION,
        )
    database_commit = {
        "committed_at": _source_file_delete_now(),
        "deleted_chunk_count": len(changed_chunk_ids),
        "deleted_chunk_ids_sample": sorted(changed_chunk_ids)[:64],
        "deleted_chunk_ids_truncated_count": max(0, len(changed_chunk_ids) - 64),
        "deleted_source_file_count": len(source_files),
        "active_chunk_scope_hash_after": active_scope_hash_after,
        "knowledge_base_chunk_version_after": int(
            knowledge_base.current_chunk_version or 0
        ),
        "chunk_version_state_id": version_state.id if version_state is not None else None,
        "chunk_version_state_hash": (
            version_state.state_hash if version_state is not None else None
        ),
        "context_graph_state_id": freshness["context_graph_state_id"],
        "freshness_rows_marked_stale": freshness["freshness_rows_marked_stale"],
        "cache_invalidation_required": True,
        "cache_invalidation_dispatched_at": None,
        # The durable Qdrant cleanup lifecycle owns point removal. This deletion closes the
        # PostgreSQL active scope and makes the old graph inadmissibly stale;
        # it must never claim that derived vector bytes were removed here.
        "qdrant_cleanup_performed": False,
    }
    committed = _transition_source_file_delete_payload(
        {**payload, "database_commit": database_commit},
        "database_committed",
    )
    row.payload_json = committed
    row.status = "cache_invalidation_pending"
    row.error_message = None
    db.flush()


def _verify_source_file_delete_database_commit(
    db: Session,
    *,
    row: IngestionCompensationLog,
    knowledge_base: KnowledgeBase,
    payload: dict[str, Any],
    target: Path,
) -> None:
    document, document_state, _document_chunks = _source_file_delete_document_state(
        db,
        knowledge_base_id=knowledge_base.id,
        source_path=str(target),
        for_update=False,
    )
    before = payload["document_before"]
    if before["exists"]:
        if (
            document is None
            or document.id != before["document_id"]
            or document_state["source_path"] != before["source_path"]
            or document_state["checksum"] != before["checksum"]
            or document_state["is_active"]
            or document_state["active_chunk_count"] != 0
            or document_state["active_chunk_scope_hash"]
            != compute_chunk_scope_hash([])
        ):
            raise SourceFileDeleteIntegrityError(
                "Source deletion database postcondition no longer holds"
            )
    elif document is not None:
        raise SourceFileDeleteIntegrityError(
            "An unplanned document appeared after source deletion commit"
        )
    _active_source_files, active_source_file_state = _source_file_delete_source_file_state(
        db,
        knowledge_base_id=knowledge_base.id,
        source_path=str(target),
        for_update=False,
    )
    if active_source_file_state:
        raise SourceFileDeleteIntegrityError(
            "SourceFile lifecycle postcondition no longer holds"
        )
    for source_file_before in payload["source_files_before"]:
        source_file = db.get(
            SourceFile,
            source_file_before["source_file_id"],
            populate_existing=True,
        )
        if (
            source_file is None
            or source_file.knowledge_base_id != knowledge_base.id
            or source_file.source_path != source_file_before["source_path"]
            or source_file.document_id != source_file_before["document_id"]
            or source_file.checksum != source_file_before["checksum"]
            or source_file.source_type != source_file_before["source_type"]
            or int(source_file.size_bytes or 0) != source_file_before["size_bytes"]
            or source_file.state != "deleted"
        ):
            raise SourceFileDeleteIntegrityError(
                "Frozen SourceFile lifecycle witness changed before cache invalidation completed"
            )
    remaining_chunks = list(
        db.scalars(
            active_chunks_query(knowledge_base.id).execution_options(
                populate_existing=True
            )
        ).all()
    )
    if compute_chunk_scope_hash(remaining_chunks) != payload["database_commit"][
        "active_chunk_scope_hash_after"
    ]:
        raise SourceFileDeleteIntegrityError(
            "Knowledge-base active scope changed before cache invalidation completed"
        )
    state_id = payload["database_commit"].get("chunk_version_state_id")
    expected_chunk_version = int(
        payload["database_commit"]["knowledge_base_chunk_version_after"]
    )
    if int(knowledge_base.current_chunk_version or 0) != expected_chunk_version:
        raise SourceFileDeleteIntegrityError(
            "Knowledge-base current chunk version changed before cache invalidation completed"
        )
    active_version_states = list(
        db.scalars(
            select(ChunkVersion).where(
                ChunkVersion.knowledge_base_id == knowledge_base.id,
                ChunkVersion.state == "active",
            ).execution_options(populate_existing=True)
        ).all()
    )
    if state_id is not None:
        version_state = db.get(ChunkVersion, state_id, populate_existing=True)
        if (
            version_state is None
            or version_state.state != "active"
            or version_state.chunk_version != expected_chunk_version
            or len(active_version_states) != 1
            or active_version_states[0].id != state_id
            or version_state.state_hash
            != payload["database_commit"].get("chunk_version_state_hash")
            or (version_state.stats_json or {}).get("active_chunk_scope_hash")
            != payload["database_commit"]["active_chunk_scope_hash_after"]
        ):
            raise SourceFileDeleteIntegrityError(
                "ChunkVersion witness changed before cache invalidation completed"
            )
    elif active_version_states:
        raise SourceFileDeleteIntegrityError(
            "Empty knowledge base retained an active ChunkVersion state"
        )
    context_state_id = payload["database_commit"].get("context_graph_state_id")
    expected_freshness_count = int(
        payload["database_commit"].get("freshness_rows_marked_stale") or 0
    )
    if context_state_id is not None:
        from app.models import ContextGraphFreshness

        freshness_rows = list(
            db.scalars(
                select(ContextGraphFreshness).where(
                    ContextGraphFreshness.knowledge_base_id == knowledge_base.id,
                    ContextGraphFreshness.context_graph_state_id == context_state_id,
                ).execution_options(populate_existing=True)
            ).all()
        )
        if (
            len(freshness_rows) != expected_freshness_count
            or any(
                not row.is_stale
                or "active_chunk_scope_changed" not in (row.stale_reasons_json or [])
                for row in freshness_rows
            )
        ):
            raise SourceFileDeleteIntegrityError(
                "Context graph freshness witness changed before cache invalidation completed"
            )
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"]
    if _source_file_delete_file_state(target, storage_root)["exists"]:
        raise SourceFileDeleteIntegrityError(
            "Source file reappeared before cache invalidation completed"
        )


def remove_knowledge_base_file(
    db: Session,
    knowledge_base: KnowledgeBase,
    source_path: str,
) -> bool:
    knowledge_base_id = str(knowledge_base.id)
    knowledge_base_name = str(knowledge_base.name)
    lease = _require_source_file_delete_resource_lock(knowledge_base_id)
    if db.new or db.dirty or db.deleted:
        raise SourceFileDeleteError(
            "Source deletion requires a clean session before opening its explicit transactions"
        )
    if db.in_transaction():
        db.commit()

    with db.begin():
        locked_knowledge_base = db.scalar(
            select(KnowledgeBase)
            .where(KnowledgeBase.id == knowledge_base_id)
            .execution_options(populate_existing=True)
            .with_for_update()
        )
        if locked_knowledge_base is None or locked_knowledge_base.name != knowledge_base_name:
            raise SourceFileDeleteConflict("Knowledge base changed before source deletion")
        active_intents = pending_source_file_delete_intents(db, knowledge_base_id)
        if lease.batch_id is not None:
            if len(active_intents) != 1 or active_intents[0].id != lease.batch_id:
                raise SourceFileDeleteConflict(
                    "Source deletion recovery owner no longer matches the active intent"
                )
            row = active_intents[0]
            payload, target = _validate_source_file_delete_payload(
                row,
                locked_knowledge_base,
            )
            requested_target = _source_file_delete_target(
                locked_knowledge_base,
                source_path,
            )
            if requested_target != target:
                raise SourceFileDeleteConflict(
                    "Source deletion recovery target does not match the durable intent"
                )
        else:
            if active_intents:
                raise SourceFileDeleteConflict(
                    "A durable source deletion intent is already active"
                )
            row = _prepare_source_file_delete_intent(
                db,
                knowledge_base=locked_knowledge_base,
                source_path=source_path,
            )
            if row is None:
                return False
            payload, target = _validate_source_file_delete_payload(
                row,
                locked_knowledge_base,
            )
        intent_id = row.id

    if payload["phase"] == "intent_committed":
        try:
            external_result = _apply_source_file_delete_external_effect(
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                knowledge_base_source_root=str(
                    locked_knowledge_base.source_root
                ),
                target=target,
                expected=payload["file_before"],
            )
        except Exception as exc:
            try:
                _record_source_file_delete_failure(db, intent_id=intent_id, exc=exc)
            except Exception as record_exc:
                raise SourceFileDeleteRecoveryError(
                    "Source deletion failed and its recovery state could not be updated"
                ) from record_exc
            if isinstance(
                exc,
                (SourceFileDeleteIntegrityError, SourceFileDeleteConflict),
            ):
                raise SourceFileDeleteRecoveryError(
                    "Source deletion target identity requires manual reconciliation"
                ) from exc
            raise SourceFileDeleteRecoveryError(
                "Source deletion did not complete; its durable intent remains recoverable"
            ) from exc
        with db.begin():
            row = _load_source_file_delete_intent(db, intent_id)
            locked_knowledge_base = db.get(
                KnowledgeBase,
                knowledge_base_id,
                populate_existing=True,
            )
            if locked_knowledge_base is None:
                raise SourceFileDeleteIntegrityError(
                    "Knowledge base disappeared during source deletion"
                )
            payload, target = _validate_source_file_delete_payload(
                row,
                locked_knowledge_base,
            )
            external_applied = _transition_source_file_delete_payload(
                payload,
                "external_applied",
                details=external_result,
            )
            row.payload_json = external_applied
            row.status = "external_applied"
            row.error_message = None
        payload = external_applied

    if payload["phase"] == "external_applied":
        try:
            with db.begin():
                row = _load_source_file_delete_intent(db, intent_id)
                locked_knowledge_base = db.scalar(
                    select(KnowledgeBase)
                    .where(KnowledgeBase.id == knowledge_base_id)
                    .execution_options(populate_existing=True)
                    .with_for_update()
                )
                if locked_knowledge_base is None:
                    raise SourceFileDeleteIntegrityError(
                        "Knowledge base disappeared during source deletion finalization"
                    )
                payload, target = _validate_source_file_delete_payload(
                    row,
                    locked_knowledge_base,
                )
                _finalize_source_file_delete_database(
                    db,
                    row=row,
                    knowledge_base=locked_knowledge_base,
                    payload=payload,
                    target=target,
                )
                payload = dict(row.payload_json or {})
        except (SourceFileDeleteIntegrityError, SourceFileDeleteConflict) as exc:
            db.rollback()
            try:
                _record_source_file_delete_failure(db, intent_id=intent_id, exc=exc)
            except Exception as record_exc:
                raise SourceFileDeleteRecoveryError(
                    "Source bytes were deleted, but deterministic database drift could not be recorded"
                ) from record_exc
            raise SourceFileDeleteRecoveryError(
                "Source bytes were deleted, but database identity drift requires manual reconciliation"
            ) from exc
        except Exception as exc:
            db.rollback()
            # ``external_applied`` was committed before this transaction.  It
            # remains the durable recovery fence for an exact-owner retry.
            raise SourceFileDeleteRecoveryError(
                "Source bytes were deleted, but database finalization remains recoverable"
            ) from exc

    if payload["phase"] != "database_committed":
        raise SourceFileDeleteIntegrityError(
            f"Unsupported source deletion recovery phase: {payload['phase']}"
        )

    try:
        with db.begin():
            row = _load_source_file_delete_intent(db, intent_id)
            locked_knowledge_base = db.get(
                KnowledgeBase,
                knowledge_base_id,
                populate_existing=True,
            )
            if locked_knowledge_base is None:
                raise SourceFileDeleteIntegrityError(
                    "Knowledge base disappeared before cache invalidation"
                )
            payload, target = _validate_source_file_delete_payload(
                row,
                locked_knowledge_base,
            )
            _verify_source_file_delete_database_commit(
                db,
                row=row,
                knowledge_base=locked_knowledge_base,
                payload=payload,
                target=target,
            )
        invalidate_context_graph_cache_after_commit(knowledge_base_id, strict=True)
    except (SourceFileDeleteIntegrityError, SourceFileDeleteConflict) as exc:
        try:
            _record_source_file_delete_failure(db, intent_id=intent_id, exc=exc)
        except Exception as record_exc:
            raise SourceFileDeleteRecoveryError(
                "Source deletion committed, but witness drift could not be recorded"
            ) from record_exc
        raise SourceFileDeleteRecoveryError(
            "Source deletion committed, but database witness drift requires manual reconciliation"
        ) from exc
    except Exception as exc:
        try:
            _record_source_file_delete_failure(db, intent_id=intent_id, exc=exc)
        except Exception as record_exc:
            raise SourceFileDeleteRecoveryError(
                "Source deletion committed, but cache recovery state could not be updated"
            ) from record_exc
        raise SourceFileDeleteRecoveryError(
            "Source deletion facts committed, but cache invalidation remains pending"
        ) from exc

    try:
        with db.begin():
            row = _load_source_file_delete_intent(db, intent_id)
            locked_knowledge_base = db.get(
                KnowledgeBase,
                knowledge_base_id,
                populate_existing=True,
            )
            if locked_knowledge_base is None:
                raise SourceFileDeleteIntegrityError(
                    "Knowledge base disappeared while completing source deletion"
                )
            payload, _target = _validate_source_file_delete_payload(
                row,
                locked_knowledge_base,
            )
            database_commit = {
                **payload["database_commit"],
                "cache_invalidation_dispatched_at": _source_file_delete_now(),
            }
            completed = _transition_source_file_delete_payload(
                {**payload, "database_commit": database_commit},
                "completed",
            )
            row.payload_json = completed
            row.status = "committed"
            row.error_message = None
    except (SourceFileDeleteIntegrityError, SourceFileDeleteConflict) as exc:
        db.rollback()
        try:
            _record_source_file_delete_failure(db, intent_id=intent_id, exc=exc)
        except Exception as record_exc:
            raise SourceFileDeleteRecoveryError(
                "Cache was invalidated, but completion witness drift could not be recorded"
            ) from record_exc
        raise SourceFileDeleteRecoveryError(
            "Cache was invalidated, but completion witness drift requires manual reconciliation"
        ) from exc
    except Exception as exc:
        db.rollback()
        raise SourceFileDeleteRecoveryError(
            "Cache was invalidated, but deletion completion audit remains recoverable"
        ) from exc
    return True


def get_job_status(db: Session, job_id: str) -> dict | None:
    job = db.get(IngestionJob, job_id)
    if job is None:
        return None
    return {
        "job_id": job.id,
        "state": job.status,
        "error": job.error_message,
        "document_id": job.document_id,
        "source_path": job.source_path,
        "batch_id": job.batch_id,
        "stats": job.stats or {},
    }

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from sqlalchemy import inspect, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import IngestionCompensationLog, KnowledgeBase
from app.services.ingestion_resource_lock import active_ingestion_resource_lease
from app.services.storage import (
    ALLOWED_UPLOAD_SUFFIXES,
    UploadValidationError,
    build_storage_path,
    compute_checksum,
    contained_path,
    durable_replace,
    durable_sync_directory,
    durable_unlink,
    normalize_upload_filename,
    normalize_upload_source_slot_key,
    namespace_durability_protocol,
    plan_storage_path,
    run_bounded_source_io,
    validate_upload_admission,
    write_upload_candidate,
)


UPLOAD_SOURCE_REPLACEMENT_OPERATION = "upload_source_replacement"
UPLOAD_SOURCE_REPLACEMENT_PROTOCOL_VERSION = "upload_source_replacement_v2"
# The v2 wire format is immutable.  In particular, do not derive its schema
# hash from the process-wide allowlist: adding a protocol for a future platform
# must not make an already-committed v2 recovery intent unreadable.
UPLOAD_SOURCE_REPLACEMENT_V2_NAMESPACE_DURABILITY_PROTOCOLS = frozenset(
    {
        "posix_parent_directory_fsync_v1",
        "windows_volume_flush_v1",
        "windows_pytest_adapter_v1",
    }
)
UPLOAD_SOURCE_REPLACEMENT_ACTIVE_STATUSES = frozenset(
    {"pending", "cleanup_pending", "rollback_pending", "manual_review"}
)
UPLOAD_SOURCE_REPLACEMENT_PHASES = frozenset(
    {
        "intent_committed",
        "candidate_ready",
        "target_backed_up",
        "candidate_installed",
        "database_committed",
        "cleanup_pending",
        "completed",
        "rollback_pending",
        "rolled_back",
        "manual_review",
    }
)
UPLOAD_SOURCE_REPLACEMENT_STATUS_PHASES = {
    "pending": frozenset(
        {
            "intent_committed",
            "candidate_ready",
            "target_backed_up",
            "candidate_installed",
        }
    ),
    "cleanup_pending": frozenset({"database_committed", "cleanup_pending"}),
    "rollback_pending": frozenset(
        {
            "intent_committed",
            "candidate_ready",
            "target_backed_up",
            "candidate_installed",
        }
    ),
    # Recovery failures preserve the last trustworthy semantic phase.  A
    # terminal phase is intentionally excluded: terminal status and phase must
    # agree exactly before an intent can leave the active recovery set.
    "manual_review": frozenset(
        {
            "intent_committed",
            "candidate_ready",
            "target_backed_up",
            "candidate_installed",
            "database_committed",
            "cleanup_pending",
            "manual_review",
            "rollback_pending",
        }
    ),
    "completed": frozenset({"completed"}),
    "rolled_back": frozenset({"rolled_back"}),
}
UPLOAD_REPLACEMENT_RECOVERY_HEALTH_PROTOCOL_VERSION = (
    "upload_replacement_recovery_health_v1"
)
_UPLOAD_REPLACEMENT_RECOVERY_HEALTH_LOCK = RLock()
_UPLOAD_REPLACEMENT_RECOVERY_HEALTH: dict[str, Any] = {
    "protocol_version": UPLOAD_REPLACEMENT_RECOVERY_HEALTH_PROTOCOL_VERSION,
    "status": "not_run",
    "last_run_at": None,
    "knowledge_bases": 0,
    "selected": 0,
    "completed": 0,
    "rolled_back": 0,
    "cleanup_pending": 0,
    "manual_review": 0,
    "failed": 0,
    "retryable": True,
}
_PAYLOAD_FIELDS = (
    "protocol_version",
    "namespace_durability_protocol",
    "schema_hash",
    "intent_id",
    "knowledge_base_id",
    "knowledge_base_name",
    "phase",
    "target_path",
    "candidate_path",
    "backup_path",
    "target_before",
    "candidate",
    "backup",
    "registration",
    "history",
    "created_at",
    "updated_at",
    "intent_payload_hash",
)
_FILE_STATE_FIELDS = ("exists", "checksum", "size_bytes")
_CANDIDATE_FIELDS = ("checksum", "size_bytes")
_REGISTRATION_FIELDS = (
    "document_id",
    "job_id",
    "candidate_checksum",
    "candidate_size_bytes",
    "committed_at",
)
_HISTORY_FIELDS = ("from", "to", "at", "details")


class UploadReplacementError(RuntimeError):
    """Base error for the durable mutable upload-slot protocol."""


class UploadReplacementIntegrityError(UploadReplacementError):
    """Raised when a durable intent or its filesystem state is untrusted."""


class UploadReplacementRecoveryError(UploadReplacementError):
    """Raised after a rollback/reconcile failure has been durably recorded."""


@dataclass(frozen=True)
class DurableUploadReplacement:
    intent_id: str
    target: Path
    checksum: str
    size_bytes: int
    logical_source_slot_key: str = ""
    display_filename: str = ""
    content_validation: dict[str, object] | None = None


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def upload_source_replacement_schema_hash() -> str:
    return _canonical_hash(
        {
            "protocol_version": UPLOAD_SOURCE_REPLACEMENT_PROTOCOL_VERSION,
            "namespace_durability_protocol_allowlist": sorted(
                UPLOAD_SOURCE_REPLACEMENT_V2_NAMESPACE_DURABILITY_PROTOCOLS
            ),
            "payload_fields": list(_PAYLOAD_FIELDS),
            "file_state_fields": list(_FILE_STATE_FIELDS),
            "candidate_fields": list(_CANDIDATE_FIELDS),
            "registration_fields": list(_REGISTRATION_FIELDS),
            "history_fields": list(_HISTORY_FIELDS),
            "phase_allowlist": sorted(UPLOAD_SOURCE_REPLACEMENT_PHASES),
            "path_protocol": "absolute_lexical_contained_nofollow_intent_siblings_v2",
            "checksum_protocol": "sha256_streaming_v1",
            "database_commit_protocol": "registration_and_phase_same_transaction_v1",
        }
    )


def _payload_hash(payload: dict[str, Any]) -> str:
    return _canonical_hash({key: value for key, value in payload.items() if key != "intent_payload_hash"})


def _refresh_payload_hash(payload: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(payload)
    refreshed["intent_payload_hash"] = _payload_hash(refreshed)
    return refreshed


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {str(exc)[:1500]}"


def _require_resource_lock(knowledge_base_id: str) -> None:
    if active_ingestion_resource_lease(knowledge_base_id) is None:
        raise UploadReplacementError(
            "Upload replacement requires the knowledge-base ingestion resource lock"
        )


def _file_state(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise UploadReplacementIntegrityError(f"Upload replacement path is a symbolic link: {path}")
    if not path.exists():
        return {"exists": False, "checksum": None, "size_bytes": 0}
    if not path.is_file():
        raise UploadReplacementIntegrityError(f"Upload replacement path is not a regular file: {path}")
    before = path.stat()
    checksum = compute_checksum(path)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise UploadReplacementIntegrityError(f"Upload replacement path changed during hashing: {path}")
    return {"exists": True, "checksum": checksum, "size_bytes": int(after.st_size)}


def _same_file_state(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        bool(actual.get("exists")) == bool(expected.get("exists"))
        and actual.get("checksum") == expected.get("checksum")
        and int(actual.get("size_bytes") or 0) == int(expected.get("size_bytes") or 0)
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _candidate_state(payload: dict[str, Any]) -> dict[str, Any] | None:
    candidate = dict(payload.get("candidate") or {})
    checksum = candidate.get("checksum")
    size_bytes = candidate.get("size_bytes")
    if not _is_sha256(checksum) or not isinstance(size_bytes, int) or size_bytes < 0:
        return None
    return {"exists": True, "checksum": checksum, "size_bytes": size_bytes}


def _validate_payload(
    row: IngestionCompensationLog,
    knowledge_base: KnowledgeBase,
) -> tuple[dict[str, Any], Path, Path, Path]:
    payload = dict(row.payload_json or {})
    if set(payload) != set(_PAYLOAD_FIELDS):
        raise UploadReplacementIntegrityError(
            f"Upload replacement intent {row.id} has an unexpected payload schema"
        )
    if payload.get("protocol_version") != UPLOAD_SOURCE_REPLACEMENT_PROTOCOL_VERSION:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has an unknown protocol")
    if (
        payload.get("namespace_durability_protocol")
        not in UPLOAD_SOURCE_REPLACEMENT_V2_NAMESPACE_DURABILITY_PROTOCOLS
    ):
        raise UploadReplacementIntegrityError(
            f"Upload replacement intent {row.id} has an unknown namespace durability protocol"
        )
    if payload.get("schema_hash") != upload_source_replacement_schema_hash():
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has an unknown schema hash")
    if payload.get("intent_id") != row.id or payload.get("knowledge_base_id") != row.knowledge_base_id:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} identity does not match its row")
    if payload.get("knowledge_base_name") != knowledge_base.name:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} knowledge-base name changed")
    if payload.get("phase") not in UPLOAD_SOURCE_REPLACEMENT_PHASES:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has an invalid phase")
    if row.status not in {
        "pending",
        "cleanup_pending",
        "rollback_pending",
        "manual_review",
        "completed",
        "rolled_back",
    }:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has an invalid status")
    if payload["phase"] not in UPLOAD_SOURCE_REPLACEMENT_STATUS_PHASES[row.status]:
        raise UploadReplacementIntegrityError(
            f"Upload replacement intent {row.id} status/phase combination is invalid: "
            f"{row.status}/{payload['phase']}"
        )
    if not all(isinstance(payload.get(field), str) and payload[field] for field in ("created_at", "updated_at")):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid timestamps")
    if payload.get("intent_payload_hash") != _payload_hash(payload):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} payload hash failed")

    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"].resolve()
    raw_target = Path(str(payload.get("target_path") or ""))
    raw_candidate = Path(str(payload.get("candidate_path") or ""))
    raw_backup = Path(str(payload.get("backup_path") or ""))
    if not all(path.is_absolute() for path in (raw_target, raw_candidate, raw_backup)):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} paths must be absolute")
    try:
        target = contained_path(raw_target, storage_root)
        candidate = contained_path(raw_candidate, storage_root)
        backup = contained_path(raw_backup, storage_root)
        safe_name = normalize_upload_filename(target.name)
    except UploadValidationError as exc:
        raise UploadReplacementIntegrityError(
            f"Upload replacement intent {row.id} path containment failed"
        ) from exc
    if Path(safe_name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has a disallowed target")
    if candidate != target.with_name(f".{target.name}.{row.id}.candidate"):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} candidate path is not intent-owned")
    if backup != target.with_name(f".{target.name}.{row.id}.backup"):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} backup path is not intent-owned")
    if list(row.target_ids_json or []) != [str(target), str(candidate), str(backup)]:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} target index does not match")

    target_before = payload.get("target_before")
    if not isinstance(target_before, dict) or set(target_before) != set(_FILE_STATE_FIELDS):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid before-state")
    if bool(target_before["exists"]):
        checksum = target_before.get("checksum")
        size_bytes = target_before.get("size_bytes")
        if (
            not _is_sha256(checksum)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid prior checksum")
    elif target_before.get("checksum") is not None or int(target_before.get("size_bytes") or 0) != 0:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid absent before-state")
    candidate_state = payload.get("candidate")
    if not isinstance(candidate_state, dict) or set(candidate_state) != set(_CANDIDATE_FIELDS):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid candidate state")
    if candidate_state["checksum"] is None:
        if candidate_state["size_bytes"] is not None or payload["phase"] != "intent_committed":
            raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has incomplete candidate facts")
    elif _candidate_state(payload) is None:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid candidate facts")
    backup_state = payload.get("backup")
    if not isinstance(backup_state, dict) or backup_state != target_before:
        raise UploadReplacementIntegrityError(
            f"Upload replacement intent {row.id} backup facts do not match the prior target bytes"
        )
    registration = payload.get("registration")
    if not isinstance(registration, dict) or set(registration) != set(_REGISTRATION_FIELDS):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid registration state")
    if payload["phase"] in {"database_committed", "cleanup_pending", "completed"}:
        if not all(
            isinstance(registration.get(key), str) and registration[key]
            for key in ("document_id", "job_id", "candidate_checksum", "committed_at")
        ):
            raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has incomplete registration facts")
        if (
            registration["candidate_checksum"] != candidate_state["checksum"]
            or registration["candidate_size_bytes"] != candidate_state["size_bytes"]
        ):
            raise UploadReplacementIntegrityError(
                f"Upload replacement intent {row.id} immutable registration witness does not match candidate facts"
            )
    elif any(registration.get(field) is not None for field in _REGISTRATION_FIELDS):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} registered before its commit phase")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) > 32:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid phase history")
    for event in history:
        if (
            not isinstance(event, dict)
            or not {"from", "to", "at"}.issubset(event)
            or not set(event).issubset(_HISTORY_FIELDS)
            or event.get("from") not in UPLOAD_SOURCE_REPLACEMENT_PHASES
            or event.get("to") not in UPLOAD_SOURCE_REPLACEMENT_PHASES
            or not isinstance(event.get("at"), str)
        ):
            raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} has invalid history event")
    return payload, target, candidate, backup


def _transition_payload(
    payload: dict[str, Any],
    phase: str,
    *,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in UPLOAD_SOURCE_REPLACEMENT_PHASES:
        raise ValueError(f"Unsupported upload replacement phase: {phase}")
    now = _utcnow()
    history = list(payload.get("history") or [])
    history.append(
        {
            "from": payload.get("phase"),
            "to": phase,
            "at": now,
            **({"details": details} if details else {}),
        }
    )
    return _refresh_payload_hash({**payload, "phase": phase, "updated_at": now, "history": history[-32:]})


def _commit_transition(
    db: Session,
    row: IngestionCompensationLog,
    payload: dict[str, Any],
    *,
    status: str,
    error_message: str | None = None,
) -> None:
    row.payload_json = payload
    row.status = status
    row.error_message = error_message
    db.commit()


def _load_intent(db: Session, intent_id: str, *, for_update: bool = True) -> IngestionCompensationLog:
    query = select(IngestionCompensationLog).where(
        IngestionCompensationLog.id == intent_id,
        IngestionCompensationLog.operation == UPLOAD_SOURCE_REPLACEMENT_OPERATION,
    )
    if for_update:
        query = query.with_for_update()
    row = db.scalar(query)
    if row is None:
        raise UploadReplacementIntegrityError(f"Upload replacement intent does not exist: {intent_id}")
    return row


def _record_recovery_failure(
    db: Session,
    intent_id: str,
    exc: BaseException,
    *,
    phase: str = "manual_review",
) -> None:
    db.rollback()
    row = _load_intent(db, intent_id)
    knowledge_base = db.get(KnowledgeBase, row.knowledge_base_id)
    if knowledge_base is None:
        raise UploadReplacementRecoveryError(
            f"Cannot persist recovery failure for upload intent {intent_id}: knowledge base is missing"
        ) from exc
    try:
        payload, _target, _candidate, _backup = _validate_payload(row, knowledge_base)
    except UploadReplacementIntegrityError:
        # Preserve the untrusted bytes for forensic review; status and the
        # bounded error field are still safe durable facts.
        row.status = phase
        row.error_message = _safe_error(exc)
        db.commit()
        return
    # Never overwrite the semantic commit phase with an error phase.  Recovery
    # must continue to distinguish an atomically committed registration from an
    # uncommitted filesystem candidate on the next retry.
    failed = _transition_payload(
        payload,
        str(payload["phase"]),
        details={
            "recovery_status": phase,
            "error_type": exc.__class__.__name__,
            "error": str(exc)[:1000],
        },
    )
    _commit_transition(db, row, failed, status=phase, error_message=_safe_error(exc))


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _remove_owned_file(path: Path) -> None:
    if _path_present(path):
        if path.is_symlink() or not path.is_file():
            raise UploadReplacementIntegrityError(f"Intent-owned cleanup path is not a regular file: {path}")
        durable_unlink(path)


def _registration_is_committed(
    db: Session,
    row: IngestionCompensationLog,
    payload: dict[str, Any],
) -> bool:
    if payload.get("phase") not in {"database_committed", "cleanup_pending", "completed"}:
        return False
    # The phase and immutable witness were written in the same PostgreSQL
    # transaction as register_uploaded_file(). A job can later be coalesced or
    # pruned, so mutable IngestionJob.stats is deliberately not used as proof.
    registration = dict(payload.get("registration") or {})
    if row.job_id not in {None, registration.get("job_id")}:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} job identity drifted")
    return True


def _rollback_uncommitted_intent(
    db: Session,
    row: IngestionCompensationLog,
    payload: dict[str, Any],
    target: Path,
    candidate: Path,
    backup: Path,
) -> None:
    before = dict(payload["target_before"])
    candidate_expected = _candidate_state(payload)
    actual_target = _file_state(target)
    actual_backup = _file_state(backup)

    if bool(before["exists"]):
        if _same_file_state(actual_target, before):
            pass
        elif candidate_expected is not None and _same_file_state(actual_target, candidate_expected):
            if not _same_file_state(actual_backup, before):
                raise UploadReplacementIntegrityError(
                    f"Cannot roll back upload intent {row.id}: verified prior bytes are missing"
                )
            durable_replace(backup, target)
            actual_backup = {"exists": False, "checksum": None, "size_bytes": 0}
        elif not actual_target["exists"] and _same_file_state(actual_backup, before):
            durable_replace(backup, target)
            actual_backup = {"exists": False, "checksum": None, "size_bytes": 0}
        else:
            raise UploadReplacementIntegrityError(
                f"Cannot roll back upload intent {row.id}: target bytes are ambiguous"
            )
        if actual_backup["exists"]:
            if not _same_file_state(actual_backup, before):
                raise UploadReplacementIntegrityError(
                    f"Cannot clean upload intent {row.id}: backup bytes are ambiguous"
                )
            _remove_owned_file(backup)
    else:
        if actual_target["exists"]:
            if candidate_expected is None or not _same_file_state(actual_target, candidate_expected):
                raise UploadReplacementIntegrityError(
                    f"Cannot roll back upload intent {row.id}: new target bytes are ambiguous"
                )
            durable_unlink(target)
        if actual_backup["exists"]:
            raise UploadReplacementIntegrityError(
                f"Cannot roll back upload intent {row.id}: unexpected backup exists for an absent prior target"
            )

    if _path_present(candidate):
        # This unique path was named in PostgreSQL before it was created.  In
        # the intent_committed phase its partial bytes have no accepted digest,
        # but are still safe to remove because they can never be a user target.
        _remove_owned_file(candidate)
    restored_target = _file_state(target)
    if not _same_file_state(restored_target, before):
        raise UploadReplacementIntegrityError(f"Upload intent {row.id} rollback verification failed")
    if row.status in {"manual_review", "rollback_pending"} and str(row.error_message or "").startswith(
        "DirectoryDurabilityError:"
    ):
        # A prior deletion can have succeeded before its directory durability
        # barrier failed.  Keep the intent active until that barrier is proven.
        durable_sync_directory(target.parent)
    rolled_back = _transition_payload(payload, "rolled_back")
    _commit_transition(db, row, rolled_back, status="rolled_back")


def _finalize_committed_intent(
    db: Session,
    row: IngestionCompensationLog,
    payload: dict[str, Any],
    target: Path,
    candidate: Path,
    backup: Path,
) -> bool:
    expected_candidate = _candidate_state(payload)
    if expected_candidate is None or not _same_file_state(_file_state(target), expected_candidate):
        raise UploadReplacementIntegrityError(
            f"Committed upload intent {row.id} target does not contain registered candidate bytes"
        )
    if _path_present(candidate):
        raise UploadReplacementIntegrityError(
            f"Committed upload intent {row.id} retained an unexpected candidate path"
        )
    before = dict(payload["target_before"])
    try:
        if _path_present(backup):
            if not bool(before["exists"]) or not _same_file_state(_file_state(backup), before):
                raise UploadReplacementIntegrityError(
                    f"Committed upload intent {row.id} backup bytes are ambiguous"
                )
            durable_unlink(backup)
        elif bool(before["exists"]):
            # A previous unlink may have happened before either its durability
            # barrier or the subsequent PostgreSQL phase commit failed.  Do
            # not retire the intent until the directory is flushed again.
            durable_sync_directory(backup.parent)
    except OSError as exc:
        pending = _transition_payload(
            payload,
            "cleanup_pending",
            details={"error_type": exc.__class__.__name__, "error": str(exc)[:1000]},
        )
        _commit_transition(db, row, pending, status="cleanup_pending", error_message=_safe_error(exc))
        return False
    completed = _transition_payload(payload, "completed")
    _commit_transition(db, row, completed, status="completed")
    return True


def reconcile_upload_source_replacement(db: Session, intent_id: str) -> dict[str, Any]:
    row = _load_intent(db, intent_id)
    _require_resource_lock(row.knowledge_base_id)
    knowledge_base = db.get(KnowledgeBase, row.knowledge_base_id)
    if knowledge_base is None:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} knowledge base is missing")
    try:
        payload, target, candidate, backup = _validate_payload(row, knowledge_base)
        if row.status in {"completed", "rolled_back"}:
            return {"intent_id": row.id, "status": row.status, "phase": payload["phase"]}
        if _registration_is_committed(db, row, payload):
            completed = _finalize_committed_intent(db, row, payload, target, candidate, backup)
            return {
                "intent_id": row.id,
                "status": "completed" if completed else "cleanup_pending",
                "phase": "completed" if completed else "cleanup_pending",
            }
        _rollback_uncommitted_intent(db, row, payload, target, candidate, backup)
        return {"intent_id": row.id, "status": "rolled_back", "phase": "rolled_back"}
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        try:
            _record_recovery_failure(db, intent_id, exc)
        except BaseException as record_exc:
            raise UploadReplacementRecoveryError(
                f"Upload intent {intent_id} recovery failed and its failure could not be durably recorded"
            ) from record_exc
        raise UploadReplacementRecoveryError(
            f"Upload intent {intent_id} recovery failed closed; manual review is required"
        ) from exc


def reconcile_upload_source_replacements(
    db: Session,
    knowledge_base_id: str,
) -> dict[str, Any]:
    _require_resource_lock(knowledge_base_id)
    rows = list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(
                IngestionCompensationLog.knowledge_base_id == knowledge_base_id,
                IngestionCompensationLog.operation == UPLOAD_SOURCE_REPLACEMENT_OPERATION,
                IngestionCompensationLog.status.in_(sorted(UPLOAD_SOURCE_REPLACEMENT_ACTIVE_STATUSES)),
            )
            .order_by(IngestionCompensationLog.created_at.asc(), IngestionCompensationLog.id.asc())
        ).all()
    )
    selected = [row.id for row in rows]
    results = [reconcile_upload_source_replacement(db, intent_id) for intent_id in selected]
    return {
        "knowledge_base_id": knowledge_base_id,
        "selected": len(selected),
        "completed": sum(1 for result in results if result["status"] == "completed"),
        "rolled_back": sum(1 for result in results if result["status"] == "rolled_back"),
        "cleanup_pending": sum(1 for result in results if result["status"] == "cleanup_pending"),
        "results": results,
    }


def pending_upload_replacement_blockers(
    db: Session,
    knowledge_base_id: str,
) -> list[dict[str, Any]]:
    """Read-only strict gate for KB deletion or other destructive workflows."""

    rows = list(
        db.scalars(
            select(IngestionCompensationLog)
            .where(
                IngestionCompensationLog.knowledge_base_id == knowledge_base_id,
                IngestionCompensationLog.operation == UPLOAD_SOURCE_REPLACEMENT_OPERATION,
                IngestionCompensationLog.status.in_(sorted(UPLOAD_SOURCE_REPLACEMENT_ACTIVE_STATUSES)),
            )
            .order_by(IngestionCompensationLog.created_at.asc(), IngestionCompensationLog.id.asc())
        ).all()
    )
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    blockers: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row.payload_json or {})
        validation_error: str | None = None
        if knowledge_base is None:
            validation_error = "knowledge_base_missing"
        else:
            try:
                payload, _target, _candidate, _backup = _validate_payload(row, knowledge_base)
            except BaseException as exc:
                validation_error = _safe_error(exc)
        blockers.append(
            {
                "intent_id": row.id,
                "status": row.status,
                "phase": payload.get("phase"),
                "protocol_version": payload.get("protocol_version"),
                "target_path": payload.get("target_path"),
                "validation_error": validation_error,
                "blocks_destructive_operation": True,
            }
        )
    return blockers


async def begin_upload_source_replacement(
    db: Session,
    knowledge_base: KnowledgeBase,
    upload: UploadFile,
    *,
    expected_checksum: str | None = None,
) -> DurableUploadReplacement:
    _require_resource_lock(knowledge_base.id)
    settings = get_settings()
    admission = await run_bounded_source_io(
        validate_upload_admission,
        upload,
        max_bytes=settings.upload_max_bytes,
        expected_checksum=expected_checksum,
    )
    filename = admission.filename
    logical_source_slot_key = normalize_upload_source_slot_key(filename)
    suffix = admission.suffix
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise UploadValidationError(
            f"Unsupported upload file type: {suffix or '[none]'}"
        )
    durability_protocol = await run_bounded_source_io(
        namespace_durability_protocol
    )
    if durability_protocol not in UPLOAD_SOURCE_REPLACEMENT_V2_NAMESPACE_DURABILITY_PROTOCOLS:
        raise UploadReplacementIntegrityError(
            "The current namespace durability protocol is not supported by upload replacement v2"
        )
    target = await run_bounded_source_io(
        plan_storage_path,
        filename,
        knowledge_base_source_root=knowledge_base.source_root,
    )
    # Reconcile every active intent for the knowledge base before accepting a
    # new path.  Filtering on an untrusted/corrupt target path could otherwise
    # skip a manual-review intent and make recovery fail open.
    recovery = await run_bounded_source_io(
        reconcile_upload_source_replacements,
        db,
        knowledge_base.id,
    )
    if recovery["cleanup_pending"]:
        raise UploadReplacementRecoveryError(
            "A committed upload still has backup cleanup pending; retry after filesystem recovery"
        )

    intent_id = str(uuid4())
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"].resolve()
    candidate = contained_path(target.with_name(f".{target.name}.{intent_id}.candidate"), storage_root)
    backup = contained_path(target.with_name(f".{target.name}.{intent_id}.backup"), storage_root)
    target_before = await run_bounded_source_io(_file_state, target)
    now = _utcnow()
    payload = _refresh_payload_hash(
        {
            "protocol_version": UPLOAD_SOURCE_REPLACEMENT_PROTOCOL_VERSION,
            "namespace_durability_protocol": durability_protocol,
            "schema_hash": upload_source_replacement_schema_hash(),
            "intent_id": intent_id,
            "knowledge_base_id": knowledge_base.id,
            "knowledge_base_name": knowledge_base.name,
            "phase": "intent_committed",
            "target_path": str(target),
            "candidate_path": str(candidate),
            "backup_path": str(backup),
            "target_before": target_before,
            "candidate": {
                "checksum": admission.checksum,
                "size_bytes": admission.size_bytes,
            },
            "backup": dict(target_before),
            "registration": {
                "document_id": None,
                "job_id": None,
                "candidate_checksum": None,
                "candidate_size_bytes": None,
                "committed_at": None,
            },
            "history": [],
            "created_at": now,
            "updated_at": now,
            "intent_payload_hash": "",
        }
    )
    row = IngestionCompensationLog(
        id=intent_id,
        knowledge_base_id=knowledge_base.id,
        operation=UPLOAD_SOURCE_REPLACEMENT_OPERATION,
        target_ids_json=[str(target), str(candidate), str(backup)],
        payload_json=payload,
        status="pending",
    )
    db.add(row)
    # This is deliberately the first commit: no candidate/target/backup file
    # has been created or renamed before PostgreSQL owns the recovery intent.
    db.commit()

    try:
        materialized_target = await run_bounded_source_io(
            build_storage_path,
            filename,
            knowledge_base_source_root=knowledge_base.source_root,
        )
        if materialized_target != target:
            raise UploadReplacementIntegrityError(
                "Upload target changed after its durable intent was committed"
            )
        checksum, size_bytes = await write_upload_candidate(
            upload,
            candidate,
            max_bytes=settings.upload_max_bytes,
            expected_checksum=admission.checksum,
            expected_size_bytes=admission.size_bytes,
        )
        row = _load_intent(db, intent_id)
        refreshed_knowledge_base = db.get(KnowledgeBase, knowledge_base.id)
        if refreshed_knowledge_base is None:
            raise UploadReplacementIntegrityError(
                f"Upload replacement intent {intent_id} knowledge base disappeared"
            )
        current, _target, _candidate, _backup = _validate_payload(row, refreshed_knowledge_base)
        current = {
            **current,
            "candidate": {"checksum": checksum, "size_bytes": size_bytes},
        }
        candidate_ready = _transition_payload(current, "candidate_ready")
        _commit_transition(db, row, candidate_ready, status="pending")

        if _path_present(backup):
            raise UploadReplacementIntegrityError("Intent-owned upload backup path appeared before replacement")
        if bool(target_before["exists"]):
            if not _same_file_state(
                await run_bounded_source_io(_file_state, target),
                target_before,
            ):
                raise UploadReplacementIntegrityError("Upload target changed while its resource lock was held")
            await run_bounded_source_io(durable_replace, target, backup)
            if not _same_file_state(
                await run_bounded_source_io(_file_state, backup),
                target_before,
            ):
                raise UploadReplacementIntegrityError("Upload backup failed prior-byte verification")
        elif target.exists():
            raise UploadReplacementIntegrityError("Upload target appeared while its resource lock was held")
        row = _load_intent(db, intent_id)
        backed_up = _transition_payload(dict(row.payload_json or {}), "target_backed_up")
        _commit_transition(db, row, backed_up, status="pending")

        await run_bounded_source_io(durable_replace, candidate, target)
        if not _same_file_state(
            await run_bounded_source_io(_file_state, target),
            {"exists": True, "checksum": checksum, "size_bytes": size_bytes},
        ):
            raise UploadReplacementIntegrityError("Installed upload candidate failed checksum verification")
        row = _load_intent(db, intent_id)
        installed = _transition_payload(dict(row.payload_json or {}), "candidate_installed")
        _commit_transition(db, row, installed, status="pending")
        return DurableUploadReplacement(
            intent_id,
            target,
            checksum,
            size_bytes,
            logical_source_slot_key,
            filename,
            admission.audit_card(),
        )
    except BaseException as exc:
        db.rollback()
        try:
            await run_bounded_source_io(
                reconcile_upload_source_replacement,
                db,
                intent_id,
            )
        except UploadReplacementRecoveryError as recovery_exc:
            raise recovery_exc from exc
        raise


def mark_upload_replacement_database_committed(
    db: Session,
    replacement: DurableUploadReplacement,
    *,
    document_id: str,
    job_id: str,
) -> None:
    """Stage the terminal database fact; the caller must commit the job atomically."""

    row = _load_intent(db, replacement.intent_id)
    _require_resource_lock(row.knowledge_base_id)
    knowledge_base = db.get(KnowledgeBase, row.knowledge_base_id)
    if knowledge_base is None:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} knowledge base is missing")
    payload, target, _candidate, _backup = _validate_payload(row, knowledge_base)
    if payload["phase"] != "candidate_installed":
        raise UploadReplacementIntegrityError(
            f"Upload replacement intent {row.id} cannot register from phase {payload['phase']}"
        )
    expected = _candidate_state(payload)
    if expected is None or target != replacement.target or replacement.checksum != expected["checksum"]:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} handle does not match its payload")
    if not _same_file_state(_file_state(target), expected):
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} target changed before registration")
    committed_payload = _transition_payload(
        {
            **payload,
            "registration": {
                "document_id": document_id,
                "job_id": job_id,
                "candidate_checksum": replacement.checksum,
                "candidate_size_bytes": replacement.size_bytes,
                "committed_at": _utcnow(),
            },
        },
        "database_committed",
    )
    row.job_id = job_id
    row.payload_json = committed_payload
    row.status = "cleanup_pending"
    row.error_message = None
    db.flush()


def complete_upload_replacement_after_database_commit(
    db: Session,
    replacement: DurableUploadReplacement,
) -> dict[str, Any]:
    """Best-effort cleanup that never converts a committed registration to failure."""

    try:
        return reconcile_upload_source_replacement(db, replacement.intent_id)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        # Recovery failure is already durable.  The API response remains a
        # success because registration and database_committed were committed
        # atomically; a later startup/path reconcile will retry or surface the
        # manual-review state.
        return {
            "intent_id": replacement.intent_id,
            "status": "manual_review",
            "phase": "manual_review",
            "error": _safe_error(exc),
        }


def record_postcommit_lock_release_failure(
    db: Session,
    replacement: DurableUploadReplacement,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort audit that cannot turn a committed upload into failure."""

    try:
        db.rollback()
        row = _load_intent(db, replacement.intent_id)
        knowledge_base = db.get(KnowledgeBase, row.knowledge_base_id)
        if knowledge_base is None:
            raise UploadReplacementIntegrityError(
                f"Upload replacement intent {row.id} knowledge base is missing"
            )
        payload, _target, _candidate, _backup = _validate_payload(row, knowledge_base)
        if not _registration_is_committed(db, row, payload):
            raise UploadReplacementIntegrityError(
                f"Upload replacement intent {row.id} has no committed registration witness"
            )
        audited = _transition_payload(
            payload,
            str(payload["phase"]),
            details={
                "postcommit_lock_release_failure": {
                    key: value
                    for key, value in diagnostics.items()
                    if key not in {"query", "authorization", "api_key"}
                }
            },
        )
        row.payload_json = audited
        row.error_message = "postcommit ingestion resource lock release failed; registration remained committed"
        db.commit()
        return {"persisted": True, "intent_id": row.id}
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        db.rollback()
        return {
            "persisted": False,
            "intent_id": replacement.intent_id,
            "error": _safe_error(exc),
        }


def upload_replacement_registration_is_committed(
    db: Session,
    replacement: DurableUploadReplacement,
) -> bool:
    row = _load_intent(db, replacement.intent_id)
    _require_resource_lock(row.knowledge_base_id)
    knowledge_base = db.get(KnowledgeBase, row.knowledge_base_id)
    if knowledge_base is None:
        raise UploadReplacementIntegrityError(f"Upload replacement intent {row.id} knowledge base is missing")
    payload, _target, _candidate, _backup = _validate_payload(row, knowledge_base)
    return _registration_is_committed(db, row, payload)


def rollback_upload_replacement(
    db: Session,
    replacement: DurableUploadReplacement,
) -> dict[str, Any]:
    return reconcile_upload_source_replacement(db, replacement.intent_id)


def upload_replacement_recovery_health() -> dict[str, Any]:
    with _UPLOAD_REPLACEMENT_RECOVERY_HEALTH_LOCK:
        return dict(_UPLOAD_REPLACEMENT_RECOVERY_HEALTH)


def _publish_upload_replacement_recovery_health(
    summary: dict[str, Any],
) -> dict[str, Any]:
    health = {
        "protocol_version": UPLOAD_REPLACEMENT_RECOVERY_HEALTH_PROTOCOL_VERSION,
        "status": (
            "degraded"
            if int(summary.get("cleanup_pending") or 0)
            or int(summary.get("manual_review") or 0)
            or int(summary.get("failed") or 0)
            else "healthy"
        ),
        "last_run_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_bases": int(summary.get("knowledge_bases") or 0),
        "selected": int(summary.get("selected") or 0),
        "completed": int(summary.get("completed") or 0),
        "rolled_back": int(summary.get("rolled_back") or 0),
        "cleanup_pending": int(summary.get("cleanup_pending") or 0),
        "manual_review": int(summary.get("manual_review") or 0),
        "failed": int(summary.get("failed") or 0),
        "retryable": int(summary.get("manual_review") or 0) == 0,
    }
    with _UPLOAD_REPLACEMENT_RECOVERY_HEALTH_LOCK:
        _UPLOAD_REPLACEMENT_RECOVERY_HEALTH.clear()
        _UPLOAD_REPLACEMENT_RECOVERY_HEALTH.update(health)
    return dict(health)


async def reconcile_pending_upload_replacements_startup() -> dict[str, Any]:
    from app.db import SessionLocal
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    with SessionLocal() as discovery:
        bind = discovery.get_bind()
        if str(bind.dialect.name) == "sqlite" and not inspect(bind).has_table(
            IngestionCompensationLog.__tablename__
        ):
            # The in-memory pytest adapter can exercise a lifespan whose schema
            # setup is intentionally stubbed.  Production PostgreSQL must never
            # silently skip the durable fact table.
            summary = {
                "knowledge_bases": 0,
                "selected": 0,
                "completed": 0,
                "rolled_back": 0,
                "cleanup_pending": 0,
                "manual_review": 0,
                "failed": 0,
                "results": [],
            }
            return {
                **summary,
                **_publish_upload_replacement_recovery_health(summary),
            }
        knowledge_base_ids = list(
            discovery.scalars(
                select(IngestionCompensationLog.knowledge_base_id)
                .where(
                    IngestionCompensationLog.operation == UPLOAD_SOURCE_REPLACEMENT_OPERATION,
                    IngestionCompensationLog.status.in_(sorted(UPLOAD_SOURCE_REPLACEMENT_ACTIVE_STATUSES)),
                )
                .distinct()
                .order_by(IngestionCompensationLog.knowledge_base_id.asc())
            ).all()
        )
    results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any]] = []
    for knowledge_base_id in knowledge_base_ids:
        try:
            with SessionLocal() as db:
                async with knowledge_base_ingestion_resource_lock(
                    db,
                    knowledge_base_id,
                    operation="upload_replacement_startup_reconcile",
                ):
                    results.append(
                        await run_bounded_source_io(
                            reconcile_upload_source_replacements,
                            db,
                            knowledge_base_id,
                        )
                    )
        except Exception as exc:
            failed_results.append(
                {
                    "knowledge_base_id": knowledge_base_id,
                    "error_type": exc.__class__.__name__,
                }
            )
    summary = {
        "knowledge_bases": len(knowledge_base_ids),
        "selected": sum(int(item["selected"]) for item in results),
        "completed": sum(int(item["completed"]) for item in results),
        "rolled_back": sum(int(item["rolled_back"]) for item in results),
        "cleanup_pending": sum(int(item["cleanup_pending"]) for item in results),
        "manual_review": sum(
            1
            for item in results
            for result in item.get("results") or []
            if result.get("status") == "manual_review"
        ),
        "failed": len(failed_results),
        "results": results,
        "failures": failed_results,
    }
    return {
        **summary,
        **_publish_upload_replacement_recovery_health(summary),
    }

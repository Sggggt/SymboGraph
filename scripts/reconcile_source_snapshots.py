from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from _context_graph_maintenance import resolve_knowledge_base, session_scope, write_report


SCRIPT_NAME = "reconcile_source_snapshots"
OPERATION_NAME = "reconcile_source_snapshots"
SHA256_HEX_LENGTH = 64
READ_CHUNK_BYTES = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit active DocumentVersion source snapshots and, only with --execute, "
            "copy checksum-proven legacy sources into immutable snapshot storage."
        )
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--knowledge-base-id")
    selector.add_argument("--knowledge-base-name")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Apply proven storage_path updates. Omit for read-only dry-run. "
            "Production execution must run inside the API container against PostgreSQL."
        ),
    )
    return parser.parse_args()


def _is_sha256(value: str) -> bool:
    return len(value) == SHA256_HEX_LENGTH and all(char in "0123456789abcdef" for char in value)


def _stable_sha256(path: Path) -> tuple[str | None, dict[str, Any]]:
    """Hash one ordinary file and reject bytes that changed during the read."""

    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            return None, {"error": "not_a_regular_file"}
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(READ_CHUNK_BYTES), b""):
                digest.update(chunk)
        after = resolved.stat()
    except (OSError, RuntimeError) as exc:
        return None, {"error": exc.__class__.__name__, "message": str(exc)[:1000]}

    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_size),
        int(before.st_mtime_ns),
    )
    after_identity = (
        int(after.st_dev),
        int(after.st_ino),
        int(after.st_size),
        int(after.st_mtime_ns),
    )
    if before_identity != after_identity:
        return None, {
            "error": "file_changed_during_checksum",
            "before_size_bytes": before.st_size,
            "after_size_bytes": after.st_size,
        }
    return digest.hexdigest(), {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "device": after.st_dev,
        "inode": after.st_ino,
        "write_protected": not bool(
            stat.S_IMODE(after.st_mode)
            & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ),
    }


def _inspect_candidate(
    path_value: str | None,
    *,
    label: str,
    expected_checksum: str,
    allowed_roots: tuple[Path, ...],
    checksum_cache: dict[str, tuple[str | None, dict[str, Any]]],
) -> dict[str, Any]:
    inspection: dict[str, Any] = {
        "label": label,
        "path": str(path_value or ""),
        "exists": False,
        "is_file": False,
        "exact_checksum_match": False,
    }
    if not path_value:
        inspection["error"] = "path_missing"
        return inspection

    raw_path = Path(path_value)
    if not raw_path.is_absolute():
        inspection["error"] = "path_not_absolute"
        return inspection
    try:
        if raw_path.is_symlink():
            inspection["error"] = "symlink_not_allowed"
            inspection["is_symlink"] = True
            return inspection
        resolved = raw_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        inspection.update({"error": exc.__class__.__name__, "message": str(exc)[:1000]})
        return inspection

    resolved_roots = tuple(root.resolve() for root in allowed_roots)
    if not any(resolved == root or root in resolved.parents for root in resolved_roots):
        inspection.update(
            {
                "error": "path_outside_allowed_roots",
                "resolved_path": str(resolved),
                "allowed_roots": [str(root) for root in resolved_roots],
            }
        )
        return inspection

    inspection["resolved_path"] = str(resolved)
    inspection["exists"] = True
    try:
        inspection["is_file"] = resolved.is_file()
        inspection["is_symlink"] = raw_path.is_symlink()
    except OSError as exc:
        inspection.update({"error": exc.__class__.__name__, "message": str(exc)[:1000]})
        return inspection
    if not inspection["is_file"]:
        inspection["error"] = "not_a_regular_file"
        return inspection

    cache_key = str(resolved)
    observed_checksum, diagnostics = checksum_cache.get(cache_key, (None, {}))
    if cache_key not in checksum_cache:
        observed_checksum, diagnostics = _stable_sha256(resolved)
        checksum_cache[cache_key] = (observed_checksum, diagnostics)
    inspection.update(diagnostics)
    inspection["observed_checksum"] = observed_checksum
    inspection["exact_checksum_match"] = observed_checksum == expected_checksum
    return inspection


def _canonical_path_for_source(
    *,
    source_path: Path,
    checksum: str,
    knowledge_base_name: str,
) -> Path:
    from app.services.storage import source_snapshot_path

    return source_snapshot_path(
        source_path.name,
        checksum,
        knowledge_base_name,
        create_parents=False,
    ).resolve()


def _active_document_versions(
    db: Session,
    *,
    knowledge_base_id: str,
    for_update: bool,
) -> list[tuple[Any, Any]]:
    from sqlalchemy import select

    from app.models import Document, DocumentVersion

    statement = (
        select(DocumentVersion, Document)
        .join(Document, DocumentVersion.document_id == Document.id)
        .where(
            Document.knowledge_base_id == knowledge_base_id,
            DocumentVersion.is_active.is_(True),
        )
        .order_by(Document.id.asc(), DocumentVersion.created_at.asc(), DocumentVersion.id.asc())
    )
    if for_update:
        statement = statement.with_for_update()
    return list(db.execute(statement).all())


def _execute_backend_allowed(db: Session, *, allow_sqlite_test_adapter: bool) -> None:
    dialect = str(db.get_bind().dialect.name)
    if dialect == "postgresql":
        return
    if (
        dialect == "sqlite"
        and allow_sqlite_test_adapter
        and bool(os.getenv("PYTEST_CURRENT_TEST"))
    ):
        return
    raise RuntimeError(
        "--execute requires PostgreSQL and must run through the API container; "
        f"refusing source snapshot writes through {dialect!r}"
    )


def _base_record(document_version: Any, document: Any) -> dict[str, Any]:
    return {
        "document_version_id": document_version.id,
        "document_id": document.id,
        "version": int(document_version.version),
        "old_storage_path": str(document_version.storage_path or ""),
        "new_storage_path": None,
        "logical_source_path": str(document.source_path or ""),
        "checksum": str(document_version.checksum or "").strip().lower(),
        "action": "pending_audit",
        "blocking": False,
        "manual_review_required": False,
        "database_updated": False,
        "possible_orphan_snapshot": False,
    }


def _mark_blocking(record: dict[str, Any], reason: str, **diagnostics: Any) -> None:
    record.update(
        {
            "action": "blocking_manual_review",
            "blocking": True,
            "manual_review_required": True,
            "blocking_reason": reason,
            **diagnostics,
        }
    )


def _audit_or_stage_record(
    *,
    document_version: Any,
    document: Any,
    knowledge_base_name: str,
    execute: bool,
    checksum_cache: dict[str, tuple[str | None, dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from app.core.config import get_settings
    from app.services.storage import snapshot_source_file

    record = _base_record(document_version, document)
    checksum = record["checksum"]
    if not _is_sha256(checksum):
        _mark_blocking(record, "invalid_document_version_checksum")
        return record, None

    knowledge_base_paths = get_settings().knowledge_base_paths_for_name(
        knowledge_base_name
    )
    storage_root = Path(knowledge_base_paths["storage_root"]).resolve()
    ingestion_root = Path(knowledge_base_paths["ingestion_root"]).resolve()
    storage_inspection = _inspect_candidate(
        record["old_storage_path"],
        label="document_version.storage_path",
        expected_checksum=checksum,
        allowed_roots=(storage_root, ingestion_root),
        checksum_cache=checksum_cache,
    )
    record["storage_path_inspection"] = storage_inspection

    current_canonical_path: Path | None = None
    if record["old_storage_path"]:
        try:
            current_canonical_path = _canonical_path_for_source(
                source_path=Path(record["old_storage_path"]),
                checksum=checksum,
                knowledge_base_name=knowledge_base_name,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            record["canonical_path_error"] = {
                "type": exc.__class__.__name__,
                "message": str(exc)[:1000],
            }

    storage_resolved = storage_inspection.get("resolved_path")
    is_canonical = bool(
        current_canonical_path is not None
        and storage_resolved
        and Path(storage_resolved) == current_canonical_path
        and not storage_inspection.get("is_symlink", False)
    )
    record["canonical_storage_path"] = str(current_canonical_path) if current_canonical_path else None
    record["storage_path_is_canonical"] = is_canonical
    if is_canonical and storage_inspection["exact_checksum_match"]:
        if not storage_inspection.get("write_protected", False):
            if not execute:
                record.update(
                    {
                        "action": "would_protect_canonical_snapshot",
                        "new_storage_path": record["old_storage_path"],
                        "integrity_verified": True,
                        "filesystem_updated": False,
                    }
                )
                return record, None
            from app.services.storage import protect_immutable_file

            try:
                protect_immutable_file(Path(str(storage_inspection["resolved_path"])))
            except Exception as exc:
                _mark_blocking(
                    record,
                    "canonical_snapshot_write_protection_failed",
                    protection_error={
                        "type": exc.__class__.__name__,
                        "message": str(exc)[:1000],
                    },
                    integrity_verified=False,
                )
                return record, None
            record.update(
                {
                    "action": "protected_canonical_snapshot",
                    "new_storage_path": record["old_storage_path"],
                    "integrity_verified": True,
                    "filesystem_updated": True,
                }
            )
            return record, None
        record.update(
            {
                "action": "keep_canonical_verified",
                "new_storage_path": record["old_storage_path"],
                "integrity_verified": True,
            }
        )
        return record, None

    candidates = [storage_inspection]
    source_candidate = storage_inspection if storage_inspection["exact_checksum_match"] else None
    storage_candidate_resolved = storage_inspection.get("resolved_path")
    logical_path = record["logical_source_path"]
    logical_resolved: str | None = None
    if logical_path:
        try:
            logical_resolved = str(Path(logical_path).resolve(strict=True))
        except (OSError, RuntimeError):
            logical_resolved = None
    if source_candidate is None and logical_resolved != storage_candidate_resolved:
        logical_inspection = _inspect_candidate(
            logical_path,
            label="document.source_path",
            expected_checksum=checksum,
            allowed_roots=(storage_root,),
            checksum_cache=checksum_cache,
        )
        candidates.append(logical_inspection)
        if logical_inspection["exact_checksum_match"]:
            source_candidate = logical_inspection
    record["source_candidates"] = candidates

    if source_candidate is None:
        _mark_blocking(
            record,
            "no_source_with_exact_document_version_checksum",
            integrity_verified=False,
        )
        return record, None

    source_path = Path(str(source_candidate["resolved_path"]))
    planned_target = _canonical_path_for_source(
        source_path=source_path,
        checksum=checksum,
        knowledge_base_name=knowledge_base_name,
    )
    record.update(
        {
            "new_storage_path": str(planned_target),
            "migration_source": source_candidate["label"],
            "migration_source_path": str(source_path),
            "integrity_verified": True,
        }
    )

    target_existed_before = planned_target.exists()
    if target_existed_before:
        target_inspection = _inspect_candidate(
            str(planned_target),
            label="canonical_snapshot_target",
            expected_checksum=checksum,
            allowed_roots=(ingestion_root,),
            checksum_cache=checksum_cache,
        )
        record["canonical_target_inspection"] = target_inspection
        if not target_inspection["exact_checksum_match"]:
            _mark_blocking(
                record,
                "canonical_snapshot_target_checksum_mismatch",
                integrity_verified=False,
            )
            return record, None

    if not execute:
        record["action"] = "would_migrate_to_canonical_snapshot"
        return record, None

    try:
        frozen_snapshot = snapshot_source_file(
            source_path,
            knowledge_base_name,
            expected_checksum=checksum,
        )
        snapshot_path = frozen_snapshot.canonical_path.resolve()
        observed_checksum = frozen_snapshot.checksum
        if snapshot_path != planned_target or observed_checksum != checksum:
            raise RuntimeError(
                "snapshot_source_file returned an identity different from the preflight target"
            )
    except Exception as exc:
        _mark_blocking(
            record,
            "snapshot_copy_or_verification_failed",
            snapshot_error={"type": exc.__class__.__name__, "message": str(exc)[:1000]},
            integrity_verified=False,
        )
        return record, None

    document_version.storage_path = str(snapshot_path)
    created_snapshot = not target_existed_before and snapshot_path.exists()
    record.update(
        {
            "action": "migrated_to_canonical_snapshot",
            "database_updated": True,
            "snapshot_created": created_snapshot,
        }
    )
    return record, {
        "record": record,
        "snapshot_path": str(snapshot_path),
        "snapshot_created": created_snapshot,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: dict[str, int] = {}
    for record in records:
        action = str(record.get("action") or "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "active_document_version_count": len(records),
        "action_counts": action_counts,
        "blocking_count": sum(bool(record.get("blocking")) for record in records),
        "manual_review_count": sum(bool(record.get("manual_review_required")) for record in records),
        "database_updated_count": sum(bool(record.get("database_updated")) for record in records),
        "possible_orphan_snapshot_count": sum(
            bool(record.get("possible_orphan_snapshot")) for record in records
        ),
    }


def _reconcile_locked(
    db: Session,
    *,
    knowledge_base: Any,
    execute: bool,
) -> dict[str, Any]:
    knowledge_base_id = str(knowledge_base.id)
    knowledge_base_name = str(knowledge_base.name)
    checksum_cache: dict[str, tuple[str | None, dict[str, Any]]] = {}
    rows = _active_document_versions(
        db,
        knowledge_base_id=knowledge_base_id,
        for_update=execute,
    )
    records: list[dict[str, Any]] = []
    staged_updates: list[dict[str, Any]] = []
    for document_version, document in rows:
        record, staged_update = _audit_or_stage_record(
            document_version=document_version,
            document=document,
            knowledge_base_name=knowledge_base_name,
            execute=execute,
            checksum_cache=checksum_cache,
        )
        records.append(record)
        if staged_update is not None:
            staged_updates.append(staged_update)

    database_error: dict[str, str] | None = None
    if execute and staged_updates:
        try:
            db.flush()
            db.commit()
        except Exception as exc:
            rollback_error: dict[str, str] | None = None
            try:
                db.rollback()
            except Exception as rollback_exc:
                rollback_error = {
                    "type": rollback_exc.__class__.__name__,
                    "message": str(rollback_exc)[:1000],
                }
            database_error = {
                "type": exc.__class__.__name__,
                "message": str(exc)[:1000],
                "commit_outcome": "unknown_reconcile_by_rerun",
                "rollback_error": rollback_error,
            }
            for staged in staged_updates:
                record = staged["record"]
                record.update(
                    {
                        "action": "database_update_outcome_unknown",
                        "database_updated": False,
                        "blocking": True,
                        "manual_review_required": True,
                        "blocking_reason": "database_update_failed",
                        "possible_orphan_snapshot": bool(staged["snapshot_created"]),
                        "rerunnable": True,
                    }
                )
    else:
        db.rollback()

    summary = _summary(records)
    blocking = summary["blocking_count"] > 0
    return {
        "script": SCRIPT_NAME,
        "execute": execute,
        "mode": "execute" if execute else "dry_run",
        "status": "database_update_failed" if database_error else ("blocking" if blocking else "ok"),
        "knowledge_base_id": knowledge_base_id,
        "knowledge_base_name": knowledge_base_name,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "impact": (
            "copy checksum-proven legacy sources into immutable snapshot storage and update "
            "active document_versions.storage_path in one transaction"
            if execute
            else (
                "read source files for checksum verification only; no snapshot-storage or PostgreSQL "
                "writes (the CLI still writes its JSON report under output/)"
            )
        ),
        "blocking": blocking,
        "rerunnable": True,
        "database_error": database_error,
        "summary": summary,
        "records": records,
    }


async def reconcile_source_snapshots(
    db: Session,
    *,
    knowledge_base: Any,
    execute: bool = False,
    allow_sqlite_test_adapter: bool = False,
) -> dict[str, Any]:
    """Audit or reconcile one knowledge base without exposing a production fallback."""

    if execute:
        _execute_backend_allowed(db, allow_sqlite_test_adapter=allow_sqlite_test_adapter)
    from app.services.ingestion_resource_lock import (
        IngestionResourceLockReleaseError,
        knowledge_base_ingestion_resource_lock,
    )

    payload: dict[str, Any] | None = None
    try:
        async with knowledge_base_ingestion_resource_lock(
            db,
            knowledge_base.id,
            operation=OPERATION_NAME,
        ) as lease:
            from app.models import KnowledgeBase

            knowledge_base_id = str(knowledge_base.id)
            db.expire_all()
            current_knowledge_base = db.get(
                KnowledgeBase,
                knowledge_base_id,
                populate_existing=True,
            )
            if current_knowledge_base is None:
                raise RuntimeError(
                    f"Knowledge base disappeared after the resource lock was acquired: {knowledge_base_id}"
                )
            payload = _reconcile_locked(
                db,
                knowledge_base=current_knowledge_base,
                execute=execute,
            )
            payload["resource_lock"] = lease.diagnostics()
    except IngestionResourceLockReleaseError as exc:
        if payload is None:
            raise
        payload.update(
            {
                "status": "resource_lock_release_failed",
                "blocking": True,
                "manual_review_required": True,
                "resource_lock_release_error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc)[:1000],
                },
            }
        )
    if payload is None:
        raise RuntimeError("Source snapshot reconciliation completed without an audit payload")
    return payload


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    with session_scope() as db:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id=args.knowledge_base_id,
            knowledge_base_name=args.knowledge_base_name,
        )
        return await reconcile_source_snapshots(
            db,
            knowledge_base=knowledge_base,
            execute=args.execute,
        )


def main() -> None:
    args = parse_args()
    try:
        payload = asyncio.run(_run(args))
    except Exception as exc:
        payload = {
            "script": SCRIPT_NAME,
            "execute": args.execute,
            "mode": "execute" if args.execute else "dry_run",
            "status": "failed",
            "blocking": True,
            "rerunnable": True,
            "error": {"type": exc.__class__.__name__, "message": str(exc)[:2000]},
            "records": [],
        }
    report = write_report(SCRIPT_NAME, payload)
    print(json.dumps({"output": str(report), **payload}, ensure_ascii=False, default=str))
    if payload.get("blocking") or payload.get("status") == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

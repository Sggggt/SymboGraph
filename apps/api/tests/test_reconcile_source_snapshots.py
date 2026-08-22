from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import reconcile_source_snapshots as reconciliation  # noqa: E402


def _checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _storage_file(sample_knowledge_base, filename: str, content: bytes) -> Path:
    from app.core.config import get_settings

    storage_root = get_settings().knowledge_base_paths_for_name(
        sample_knowledge_base.name
    )["storage_root"]
    storage_root.mkdir(parents=True, exist_ok=True)
    path = storage_root / filename
    path.write_bytes(content)
    return path


def _document_version(
    db_session,
    sample_knowledge_base,
    *,
    source_path: Path,
    storage_path: Path,
    checksum: str,
    title: str,
):
    from app.models import Document, DocumentVersion

    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title=title,
        source_path=str(source_path),
        source_type="markdown",
        tags=[],
        checksum=checksum,
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=checksum,
        storage_path=str(storage_path),
        is_active=True,
    )
    db_session.add(version)
    db_session.commit()
    return document, version


def _record(payload: dict, document_version_id: str) -> dict:
    return next(
        item
        for item in payload["records"]
        if item["document_version_id"] == document_version_id
    )


@pytest.mark.asyncio
async def test_reconcile_source_snapshots_dry_run_keeps_verified_canonical_and_does_not_copy_legacy(
    tmp_path,
    db_session,
    sample_knowledge_base,
):
    from app.services.storage import snapshot_source_file, source_snapshot_path

    canonical_content = b"canonical source bytes"
    canonical_source = _storage_file(
        sample_knowledge_base,
        "canonical.md",
        canonical_content,
    )
    frozen_snapshot = snapshot_source_file(
        canonical_source,
        sample_knowledge_base.name,
    )
    canonical_path = frozen_snapshot.canonical_path
    canonical_checksum = frozen_snapshot.checksum
    _, canonical_version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=canonical_source,
        storage_path=canonical_path,
        checksum=canonical_checksum,
        title="canonical",
    )

    legacy_content = b"legacy source bytes"
    legacy_source = _storage_file(
        sample_knowledge_base,
        "legacy.md",
        legacy_content,
    )
    legacy_checksum = _checksum(legacy_content)
    _, legacy_version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=legacy_source,
        storage_path=legacy_source,
        checksum=legacy_checksum,
        title="legacy",
    )
    planned_legacy_path = source_snapshot_path(
        legacy_source.name,
        legacy_checksum,
        sample_knowledge_base.name,
        create_parents=False,
    )
    assert not planned_legacy_path.exists()

    payload = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=False,
    )

    assert payload["mode"] == "dry_run"
    assert payload["status"] == "ok"
    assert _record(payload, canonical_version.id)["action"] == "keep_canonical_verified"
    legacy_record = _record(payload, legacy_version.id)
    assert legacy_record["action"] == "would_migrate_to_canonical_snapshot"
    assert legacy_record["old_storage_path"] == str(legacy_source)
    assert legacy_record["new_storage_path"] == str(planned_legacy_path.resolve())
    assert legacy_record["checksum"] == legacy_checksum
    assert not planned_legacy_path.exists()
    db_session.expire_all()
    assert db_session.get(type(legacy_version), legacy_version.id).storage_path == str(legacy_source)


@pytest.mark.asyncio
async def test_reconcile_source_snapshots_execute_migrates_exact_legacy_and_logical_sources(
    tmp_path,
    db_session,
    sample_knowledge_base,
):
    from app.services.storage import compute_checksum

    legacy_source = _storage_file(
        sample_knowledge_base,
        "legacy-exact.md",
        b"legacy exact bytes",
    )
    legacy_checksum = compute_checksum(legacy_source)
    _, legacy_version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=legacy_source,
        storage_path=legacy_source,
        checksum=legacy_checksum,
        title="legacy exact",
    )

    logical_source = _storage_file(
        sample_knowledge_base,
        "logical-fallback.md",
        b"logical exact bytes",
    )
    logical_checksum = compute_checksum(logical_source)
    missing_storage = logical_source.parent / "missing-legacy.md"
    _, logical_version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=logical_source,
        storage_path=missing_storage,
        checksum=logical_checksum,
        title="logical fallback",
    )

    payload = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=True,
        allow_sqlite_test_adapter=True,
    )

    assert payload["mode"] == "execute"
    assert payload["status"] == "ok"
    assert payload["resource_lock"]["operation"] == reconciliation.OPERATION_NAME
    assert payload["summary"]["database_updated_count"] == 2
    for version, expected_checksum in (
        (legacy_version, legacy_checksum),
        (logical_version, logical_checksum),
    ):
        record = _record(payload, version.id)
        assert record["action"] == "migrated_to_canonical_snapshot"
        assert record["checksum"] == expected_checksum
        assert record["old_storage_path"] != record["new_storage_path"]
        snapshot_path = Path(record["new_storage_path"])
        assert snapshot_path.is_file()
        assert compute_checksum(snapshot_path) == expected_checksum
        db_session.expire_all()
        assert db_session.get(type(version), version.id).storage_path == str(snapshot_path)
    assert _record(payload, logical_version.id)["migration_source"] == "document.source_path"


@pytest.mark.asyncio
async def test_reconcile_source_snapshots_repairs_writable_canonical_snapshot(
    db_session,
    sample_knowledge_base,
):
    from app.services.storage import (
        snapshot_source_file,
        verify_immutable_file_protection,
    )

    source = _storage_file(
        sample_knowledge_base,
        "writable-canonical.md",
        b"canonical bytes needing permission repair",
    )
    frozen_snapshot = snapshot_source_file(source, sample_knowledge_base.name)
    snapshot = frozen_snapshot.canonical_path
    checksum = frozen_snapshot.checksum
    os.chmod(snapshot, stat.S_IMODE(snapshot.stat().st_mode) | stat.S_IWUSR)
    _, version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=source,
        storage_path=snapshot,
        checksum=checksum,
        title="writable canonical",
    )

    preview = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=False,
    )
    assert _record(preview, version.id)["action"] == "would_protect_canonical_snapshot"
    assert stat.S_IMODE(snapshot.stat().st_mode) & stat.S_IWUSR

    applied = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=True,
        allow_sqlite_test_adapter=True,
    )
    assert _record(applied, version.id)["action"] == "protected_canonical_snapshot"
    verify_immutable_file_protection(snapshot)


@pytest.mark.asyncio
async def test_reconcile_source_snapshots_blocks_without_any_exact_source(
    tmp_path,
    db_session,
    sample_knowledge_base,
):
    expected_checksum = _checksum(b"unavailable historical bytes")
    mismatched_storage = _storage_file(
        sample_knowledge_base,
        "wrong-storage.md",
        b"wrong storage bytes",
    )
    mismatched_logical = _storage_file(
        sample_knowledge_base,
        "wrong-logical.md",
        b"wrong logical bytes",
    )
    _, version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=mismatched_logical,
        storage_path=mismatched_storage,
        checksum=expected_checksum,
        title="manual review",
    )

    payload = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=True,
        allow_sqlite_test_adapter=True,
    )

    record = _record(payload, version.id)
    assert payload["status"] == "blocking"
    assert payload["summary"]["blocking_count"] == 1
    assert record["action"] == "blocking_manual_review"
    assert record["blocking_reason"] == "no_source_with_exact_document_version_checksum"
    assert record["manual_review_required"] is True
    assert record["new_storage_path"] is None
    db_session.expire_all()
    assert db_session.get(type(version), version.id).storage_path == str(mismatched_storage)


@pytest.mark.asyncio
async def test_reconcile_source_snapshots_rejects_out_of_root_paths_without_reading_them(
    monkeypatch,
    tmp_path,
    db_session,
    sample_knowledge_base,
):
    outside = tmp_path / "outside-kb.md"
    outside.write_bytes(b"must not be read by the reconciler")
    checksum = _checksum(outside.read_bytes())
    _, version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=outside,
        storage_path=outside,
        checksum=checksum,
        title="outside root",
    )

    def fail_if_hashed(_path: Path):
        raise AssertionError("out-of-root source must be rejected before hashing")

    monkeypatch.setattr(reconciliation, "_stable_sha256", fail_if_hashed)
    payload = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=False,
    )

    record = _record(payload, version.id)
    assert payload["status"] == "blocking"
    assert record["blocking_reason"] == "no_source_with_exact_document_version_checksum"
    assert record["storage_path_inspection"]["error"] == "path_outside_allowed_roots"
    assert record["source_candidates"][-1]["error"] == "path_outside_allowed_roots"


@pytest.mark.asyncio
async def test_reconcile_source_snapshots_reports_orphan_on_database_failure_and_is_rerunnable(
    monkeypatch,
    tmp_path,
    db_session,
    sample_knowledge_base,
):
    from app.services.storage import compute_checksum

    legacy_source = _storage_file(
        sample_knowledge_base,
        "retryable.md",
        b"retryable exact bytes",
    )
    checksum = compute_checksum(legacy_source)
    _, version = _document_version(
        db_session,
        sample_knowledge_base,
        source_path=legacy_source,
        storage_path=legacy_source,
        checksum=checksum,
        title="retryable",
    )
    original_commit = db_session.commit

    def fail_commit() -> None:
        raise RuntimeError("forced document version commit failure")

    monkeypatch.setattr(db_session, "commit", fail_commit)
    failed = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=True,
        allow_sqlite_test_adapter=True,
    )

    failed_record = _record(failed, version.id)
    assert failed["status"] == "database_update_failed"
    assert failed["database_error"]["commit_outcome"] == "unknown_reconcile_by_rerun"
    assert failed_record["action"] == "database_update_outcome_unknown"
    assert failed_record["possible_orphan_snapshot"] is True
    assert failed_record["rerunnable"] is True
    orphan_path = Path(failed_record["new_storage_path"])
    assert orphan_path.is_file()
    assert compute_checksum(orphan_path) == checksum
    db_session.expire_all()
    assert db_session.get(type(version), version.id).storage_path == str(legacy_source)

    monkeypatch.setattr(db_session, "commit", original_commit)
    retried = await reconciliation.reconcile_source_snapshots(
        db_session,
        knowledge_base=sample_knowledge_base,
        execute=True,
        allow_sqlite_test_adapter=True,
    )
    retried_record = _record(retried, version.id)
    assert retried["status"] == "ok"
    assert retried_record["action"] == "migrated_to_canonical_snapshot"
    assert retried_record["snapshot_created"] is False
    db_session.expire_all()
    assert db_session.get(type(version), version.id).storage_path == str(orphan_path)


@pytest.mark.asyncio
async def test_reconcile_source_snapshots_execute_has_no_sqlite_cli_fallback(
    db_session,
    sample_knowledge_base,
):
    with pytest.raises(RuntimeError, match="requires PostgreSQL"):
        await reconciliation.reconcile_source_snapshots(
            db_session,
            knowledge_base=sample_knowledge_base,
            execute=True,
        )

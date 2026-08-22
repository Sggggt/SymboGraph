from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import select


def _prepare_storage_roots(knowledge_base) -> Path:
    from app.core.config import get_settings

    paths = get_settings().knowledge_base_paths_for_name(knowledge_base.name)
    paths["storage_root"].mkdir(parents=True, exist_ok=True)
    paths["ingestion_root"].mkdir(parents=True, exist_ok=True)
    return paths["storage_root"]


def _add_document_chunk(db, knowledge_base, source_path: Path, *, suffix: str):
    from app.models import Chunk, Document, DocumentVersion, SourceFile
    from app.core.config import get_settings
    from app.services.storage import (
        LOGICAL_SOURCE_SLOT_PROTOCOL_VERSION,
        compute_checksum,
        source_slot_key_for_path,
    )

    text = f"# {suffix}\n\nGrounded source text for {suffix}."
    source_path.write_text(text, encoding="utf-8")
    checksum = compute_checksum(source_path)
    storage_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["storage_root"]
    logical_source_slot_key = source_slot_key_for_path(
        source_path,
        storage_root,
    )
    document = Document(
        knowledge_base_id=knowledge_base.id,
        title=suffix,
        source_path=str(source_path),
        logical_source_slot_key=logical_source_slot_key,
        source_slot_protocol_version=LOGICAL_SOURCE_SLOT_PROTOCOL_VERSION,
        source_type="markdown",
        checksum=checksum,
        is_active=True,
    )
    db.add(document)
    db.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=checksum,
        storage_path=str(source_path),
        is_active=True,
    )
    db.add(version)
    db.flush()
    chunk = Chunk(
        knowledge_base_id=knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        chunk_version=1,
        chunk_index=0,
        token_start=0,
        token_end=8,
        char_start=0,
        char_end=len(text),
        text=text,
        text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        section_path=suffix,
        metadata_json={
            "chunk_schema_version": "chunk_schema_v1",
            "tokenizer_version": "symbograph_regex_tokenizer_v1",
            "chunk_size": 512,
            "chunk_overlap": 80,
        },
        state="active",
    )
    db.add(chunk)
    db.add(
        SourceFile(
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            source_path=str(source_path),
            logical_source_slot_key=logical_source_slot_key,
            source_slot_protocol_version=LOGICAL_SOURCE_SLOT_PROTOCOL_VERSION,
            checksum=checksum,
            source_type="markdown",
            size_bytes=source_path.stat().st_size,
            state="active",
        )
    )
    db.flush()
    return document, version, chunk


def _install_active_scope_and_freshness(db, knowledge_base):
    from app.models import ContextGraphFreshness, ContextGraphState
    from app.services.context_graph import active_chunks_query, write_chunk_version_state

    knowledge_base.current_chunk_version = 1
    chunks = list(db.scalars(active_chunks_query(knowledge_base.id)).all())
    version_state = write_chunk_version_state(
        db,
        knowledge_base_id=knowledge_base.id,
        chunk_version=1,
        chunks=chunks,
        chunk_size=512,
        chunk_overlap=80,
    )
    db.flush()
    graph_state = ContextGraphState(
        knowledge_base_id=knowledge_base.id,
        chunk_scope_hash=version_state.stats_json["active_chunk_scope_hash"],
        structure_graph_hash="1" * 64,
        chunk_relation_graph_hash="2" * 64,
        rq_membership_hash="3" * 64,
        mid_concept_hash="4" * 64,
        coarse_concept_hash="5" * 64,
        context_graph_hash="6" * 64,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db.add(graph_state)
    db.flush()
    freshness_rows = []
    for layer in (
        "contextual_index",
        "chunk_structure",
        "chunk_relation",
        "mid_concepts",
        "coarse_concepts",
        "context_graph",
    ):
        row = ContextGraphFreshness(
            knowledge_base_id=knowledge_base.id,
            context_graph_state_id=graph_state.id,
            layer=layer,
            state_hash="7" * 64,
            is_stale=False,
            stale_reasons_json=[],
            diagnostics_json={},
        )
        db.add(row)
        freshness_rows.append(row)
    db.commit()
    return version_state, graph_state, freshness_rows


@pytest.mark.asyncio
@pytest.mark.parametrize("source_exists", [True, False])
async def test_file_only_source_delete_closes_source_file_lifecycle(
    monkeypatch,
    db_session,
    sample_knowledge_base,
    source_exists,
):
    from app.models import IngestionCompensationLog, SourceFile
    from app.services import ingestion
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
    from app.services.storage import compute_checksum

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    source_path = storage_root / "pending-file-only.md"
    source_path.write_text("# Pending\n\nNot parsed yet.", encoding="utf-8")
    source_file = SourceFile(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=None,
        source_path=str(source_path),
        checksum=compute_checksum(source_path),
        source_type="markdown",
        size_bytes=source_path.stat().st_size,
        state="active",
    )
    db_session.add(source_file)
    db_session.commit()
    if not source_exists:
        source_path.unlink()
    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        lambda _knowledge_base_id, *, strict=False: strict,
    )

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
    ):
        assert ingestion.remove_knowledge_base_file(
            db_session,
            sample_knowledge_base,
            str(source_path),
        ) is True

    db_session.expire_all()
    assert source_path.exists() is False
    assert db_session.get(SourceFile, source_file.id).state == "deleted"
    intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == ingestion.SOURCE_FILE_DELETE_OPERATION
        )
    )
    assert intent.status == "committed"
    assert intent.payload_json["database_commit"]["deleted_source_file_count"] == 1


@pytest.mark.asyncio
async def test_deleting_last_active_document_deactivates_version_and_resets_empty_library(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import ChunkVersion, IngestionCompensationLog
    from app.services import ingestion
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    _document, _version, _chunk = _add_document_chunk(
        db_session,
        sample_knowledge_base,
        storage_root / "last-active.md",
        suffix="last-active",
    )
    version_state, _graph_state, _freshness = _install_active_scope_and_freshness(
        db_session,
        sample_knowledge_base,
    )
    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        lambda _knowledge_base_id, *, strict=False: strict,
    )

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
    ):
        assert ingestion.remove_knowledge_base_file(
            db_session,
            sample_knowledge_base,
            str(storage_root / "last-active.md"),
        ) is True

    db_session.expire_all()
    assert sample_knowledge_base.current_chunk_version == 0
    assert db_session.get(ChunkVersion, version_state.id).state == "inactive"
    intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == ingestion.SOURCE_FILE_DELETE_OPERATION
        )
    )
    witness = intent.payload_json["database_commit"]
    assert witness["knowledge_base_chunk_version_after"] == 0
    assert witness["chunk_version_state_id"] is None
    assert witness["chunk_version_state_hash"] is None


@pytest.mark.asyncio
async def test_delete_route_rewrites_complete_scope_marks_freshness_then_invalidates_cache(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.db import SessionLocal
    from app.models import (
        Chunk,
        ChunkVersion,
        ContextGraphFreshness,
        IngestionCompensationLog,
        SourceFile,
    )
    from app.routers import knowledge
    from app.services import ingestion
    from app.services.context_graph import active_chunks_query, compute_chunk_scope_hash

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    first_document, _first_version, first_chunk = _add_document_chunk(
        db_session,
        sample_knowledge_base,
        storage_root / "delete-me.md",
        suffix="delete-me",
    )
    second_document, _second_version, second_chunk = _add_document_chunk(
        db_session,
        sample_knowledge_base,
        storage_root / "keep-me.md",
        suffix="keep-me",
    )
    _version_state, graph_state, freshness_rows = _install_active_scope_and_freshness(
        db_session,
        sample_knowledge_base,
    )
    cache_observations: list[tuple[str, str, bool]] = []

    def observe_postcommit_invalidation(knowledge_base_id: str, *, strict: bool = False) -> None:
        assert strict is True
        with SessionLocal() as verification:
            deleted = verification.get(Chunk, first_chunk.id)
            kept = verification.get(Chunk, second_chunk.id)
            cache_observations.append(
                (knowledge_base_id, deleted.state, bool(kept.state == "active"))
            )

    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        observe_postcommit_invalidation,
    )

    response = await knowledge.delete_knowledge_base_file(
        str(storage_root / "delete-me.md"),
        sample_knowledge_base.id,
        db_session,
    )

    assert response == {"removed": True}
    assert not (storage_root / "delete-me.md").exists()
    db_session.expire_all()
    assert db_session.get(type(first_document), first_document.id).is_active is False
    assert db_session.get(type(second_document), second_document.id).is_active is True
    assert db_session.get(Chunk, first_chunk.id).state == "deleted"
    assert db_session.get(Chunk, second_chunk.id).state == "active"
    deleted_source_file = db_session.scalar(
        select(SourceFile).where(SourceFile.document_id == first_document.id)
    )
    assert deleted_source_file.state == "deleted"
    remaining = list(db_session.scalars(active_chunks_query(sample_knowledge_base.id)).all())
    assert [chunk.id for chunk in remaining] == [second_chunk.id]
    version_state = db_session.scalar(
        select(ChunkVersion).where(
            ChunkVersion.knowledge_base_id == sample_knowledge_base.id,
            ChunkVersion.chunk_version == 1,
        )
    )
    assert version_state.stats_json["chunk_count"] == 1
    assert version_state.stats_json["active_chunk_scope_hash"] == compute_chunk_scope_hash(
        remaining
    )
    for freshness in freshness_rows:
        persisted = db_session.get(ContextGraphFreshness, freshness.id)
        assert persisted.is_stale is True
        assert "active_chunk_scope_changed" in persisted.stale_reasons_json
    assert db_session.get(type(graph_state), graph_state.id).diagnostics_json[
        "active_scope_stale"
    ] is True
    assert cache_observations == [(sample_knowledge_base.id, "deleted", True)]
    intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == ingestion.SOURCE_FILE_DELETE_OPERATION
        )
    )
    assert intent.status == "committed"
    assert intent.payload_json["phase"] == "completed"
    assert intent.payload_json["database_commit"]["qdrant_cleanup_performed"] is False
    assert intent.payload_json["database_commit"]["deleted_source_file_count"] == 1
    assert intent.payload_json["database_commit"]["cache_invalidation_dispatched_at"]


@pytest.mark.asyncio
async def test_unlink_uncertainty_keeps_durable_intent_and_central_kb_fence_until_exact_retry(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import Chunk, IngestionCompensationLog
    from app.services import ingestion, maintenance
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    document, _version, chunk = _add_document_chunk(
        db_session,
        sample_knowledge_base,
        storage_root / "crash-window.md",
        suffix="crash-window",
    )
    _install_active_scope_and_freshness(db_session, sample_knowledge_base)
    source_path = storage_root / "crash-window.md"
    original_unlink = ingestion.durable_unlink

    def unlink_then_report_uncertain(
        path: Path,
        *,
        missing_ok: bool = False,
        expected_identity=None,
    ) -> None:
        path.unlink(missing_ok=missing_ok)
        raise OSError("simulated fsync outcome uncertainty")

    monkeypatch.setattr(ingestion, "durable_unlink", unlink_then_report_uncertain)
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
    ):
        with pytest.raises(ingestion.SourceFileDeleteRecoveryError):
            ingestion.remove_knowledge_base_file(
                db_session,
                sample_knowledge_base,
                str(source_path),
            )

    db_session.expire_all()
    assert source_path.exists() is False
    assert db_session.get(type(document), document.id).is_active is True
    assert db_session.get(Chunk, chunk.id).state == "active"
    intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == ingestion.SOURCE_FILE_DELETE_OPERATION
        )
    )
    assert intent.status == "external_failed"
    assert intent.payload_json["phase"] == "intent_committed"

    with pytest.raises(maintenance.MaintenanceConflict, match="durable source-file deletion"):
        maintenance.delete_knowledge_base_data(db_session, sample_knowledge_base)

    with pytest.raises(IngestionResourceBusyError) as blocked:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation="context_graph_rebuild",
        ):
            raise AssertionError(
                "pending source deletion must fence every lock-coordinated KB mutation"
            )
    assert blocked.value.diagnostics["reason"] == "source_file_delete_recovery_pending"
    assert blocked.value.diagnostics["intent_id"] == intent.id

    monkeypatch.setattr(ingestion, "durable_unlink", original_unlink)
    replayed_parent_syncs: list[Path] = []
    monkeypatch.setattr(
        ingestion,
        "durable_sync_directory",
        lambda path: replayed_parent_syncs.append(path),
    )
    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        lambda _knowledge_base_id, *, strict=False: None,
    )
    recovery_owner = ingestion.source_file_delete_lock_token(
        db_session,
        sample_knowledge_base,
        str(source_path),
    )
    assert recovery_owner == intent.id
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
        batch_id=recovery_owner,
    ):
        assert ingestion.remove_knowledge_base_file(
            db_session,
            sample_knowledge_base,
            str(source_path),
        ) is True

    db_session.expire_all()
    assert db_session.get(type(document), document.id).is_active is False
    assert db_session.get(Chunk, chunk.id).state == "deleted"
    assert db_session.get(IngestionCompensationLog, intent.id).status == "committed"
    assert replayed_parent_syncs == [source_path.parent]


@pytest.mark.asyncio
async def test_deterministic_finalize_drift_is_persisted_as_manual_review(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import Chunk, IngestionCompensationLog
    from app.services import ingestion
    from app.services.ingestion_resource_lock import (
        IngestionResourceBusyError,
        knowledge_base_ingestion_resource_lock,
    )

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    document, _version, chunk = _add_document_chunk(
        db_session,
        sample_knowledge_base,
        storage_root / "manual-review.md",
        suffix="manual-review",
    )
    _install_active_scope_and_freshness(db_session, sample_knowledge_base)
    source_path = storage_root / "manual-review.md"

    def reject_version_rewrite(*_args, **_kwargs):
        raise RuntimeError("deterministic descriptor drift")

    monkeypatch.setattr(
        ingestion,
        "rewrite_chunk_version_state_for_active_scope_mutation",
        reject_version_rewrite,
    )
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
    ):
        with pytest.raises(ingestion.SourceFileDeleteRecoveryError, match="manual reconciliation"):
            ingestion.remove_knowledge_base_file(
                db_session,
                sample_knowledge_base,
                str(source_path),
            )

    db_session.expire_all()
    assert source_path.exists() is False
    assert db_session.get(type(document), document.id).is_active is True
    assert db_session.get(Chunk, chunk.id).state == "active"
    intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == ingestion.SOURCE_FILE_DELETE_OPERATION
        )
    )
    assert intent.status == "manual_review"
    assert intent.payload_json["phase"] == "external_applied"

    with pytest.raises(IngestionResourceBusyError) as blocked:
        async with knowledge_base_ingestion_resource_lock(
            db_session,
            sample_knowledge_base.id,
            operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
            batch_id=intent.id,
        ):
            raise AssertionError("manual-review intent must not be retryable")
    assert blocked.value.diagnostics["retryable"] is False


@pytest.mark.asyncio
async def test_cache_failure_keeps_postcommit_recovery_fence_and_retry_is_idempotent(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.models import Chunk, IngestionCompensationLog
    from app.services import ingestion
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    document, _version, chunk = _add_document_chunk(
        db_session,
        sample_knowledge_base,
        storage_root / "cache-retry.md",
        suffix="cache-retry",
    )
    _install_active_scope_and_freshness(db_session, sample_knowledge_base)
    source_path = storage_root / "cache-retry.md"

    def fail_cache_invalidation(_knowledge_base_id: str, *, strict: bool = False) -> None:
        assert strict is True
        raise OSError("simulated redis invalidation failure")

    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        fail_cache_invalidation,
    )
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
    ):
        with pytest.raises(ingestion.SourceFileDeleteRecoveryError, match="cache invalidation"):
            ingestion.remove_knowledge_base_file(
                db_session,
                sample_knowledge_base,
                str(source_path),
            )

    db_session.expire_all()
    assert source_path.exists() is False
    assert db_session.get(type(document), document.id).is_active is False
    assert db_session.get(Chunk, chunk.id).state == "deleted"
    intent = db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == ingestion.SOURCE_FILE_DELETE_OPERATION
        )
    )
    assert intent.status == "cache_invalidation_pending"
    assert intent.payload_json["phase"] == "database_committed"

    dispatched: list[str] = []
    monkeypatch.setattr(
        ingestion,
        "invalidate_context_graph_cache_after_commit",
        lambda knowledge_base_id, *, strict=False: (
            dispatched.append(knowledge_base_id) if strict else None
        ),
    )
    owner = ingestion.source_file_delete_lock_token(
        db_session,
        sample_knowledge_base,
        str(source_path),
    )
    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
        batch_id=owner,
    ):
        assert ingestion.remove_knowledge_base_file(
            db_session,
            sample_knowledge_base,
            str(source_path),
        ) is True
    assert dispatched == [sample_knowledge_base.id]
    db_session.expire_all()
    assert db_session.get(IngestionCompensationLog, intent.id).status == "committed"


@pytest.mark.asyncio
async def test_duplicate_document_source_identity_fails_before_unlink_or_intent(
    db_session,
    sample_knowledge_base,
):
    from app.models import Document, IngestionCompensationLog
    from app.services import ingestion
    from app.services.ingestion_resource_lock import knowledge_base_ingestion_resource_lock
    from app.services.storage import compute_checksum

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    source_path = storage_root / "duplicate-slot.md"
    source_path.write_text("duplicate logical slot", encoding="utf-8")
    checksum = compute_checksum(source_path)
    for suffix in ("a", "b"):
        db_session.add(
            Document(
                knowledge_base_id=sample_knowledge_base.id,
                title=f"duplicate-{suffix}",
                source_path=str(source_path),
                source_type="markdown",
                checksum=checksum,
                is_active=True,
            )
        )
    db_session.commit()

    async with knowledge_base_ingestion_resource_lock(
        db_session,
        sample_knowledge_base.id,
        operation=ingestion.SOURCE_FILE_DELETE_LOCK_OPERATION,
    ):
        with pytest.raises(ingestion.SourceFileDeleteConflict, match="Multiple documents"):
            ingestion.remove_knowledge_base_file(
                db_session,
                sample_knowledge_base,
                str(source_path),
            )

    assert source_path.is_file()
    assert db_session.scalar(
        select(IngestionCompensationLog).where(
            IngestionCompensationLog.operation == ingestion.SOURCE_FILE_DELETE_OPERATION
        )
    ) is None


def test_source_file_delete_service_refuses_to_mutate_without_resource_lock(
    db_session,
    sample_knowledge_base,
):
    from app.services import ingestion

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    source_path = storage_root / "lock-required.md"
    source_path.write_text("must remain", encoding="utf-8")

    with pytest.raises(ingestion.SourceFileDeleteError, match="resource lock"):
        ingestion.remove_knowledge_base_file(
            db_session,
            sample_knowledge_base,
            str(source_path),
        )
    assert source_path.is_file()


@pytest.mark.asyncio
async def test_delete_route_maps_pending_durable_failure_to_conflict(
    monkeypatch,
    db_session,
    sample_knowledge_base,
):
    from app.routers import knowledge
    from app.services import ingestion

    storage_root = _prepare_storage_roots(sample_knowledge_base)
    _document, _version, _chunk = _add_document_chunk(
        db_session,
        sample_knowledge_base,
        storage_root / "route-failure.md",
        suffix="route-failure",
    )
    _install_active_scope_and_freshness(db_session, sample_knowledge_base)

    def fail_unlink(
        _path: Path,
        *,
        missing_ok: bool = False,
        expected_identity=None,
    ) -> None:
        raise OSError("simulated unlink failure")

    monkeypatch.setattr(ingestion, "durable_unlink", fail_unlink)
    with pytest.raises(HTTPException) as raised:
        await knowledge.delete_knowledge_base_file(
            str(storage_root / "route-failure.md"),
            sample_knowledge_base.id,
            db_session,
        )
    assert raised.value.status_code == 409
    assert raised.value.detail["code"] == "source_file_delete_recovery_pending"

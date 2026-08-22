from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select


def _kb_source(knowledge_base, filename: str) -> Path:
    from app.core.config import get_settings

    storage_root = get_settings().knowledge_base_paths_for_name(
        knowledge_base.name
    )["storage_root"]
    storage_root.mkdir(parents=True, exist_ok=True)
    return (storage_root / filename).resolve()


def _metadata(document) -> dict:
    return {
        "title": document.title,
        "source_path": document.source_path,
        "source_type": document.source_type,
        "language": document.language,
        "tags": list(document.tags or []),
        "difficulty": document.difficulty,
        "checksum": document.checksum,
        "visibility": document.visibility,
        "is_active": document.is_active,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
    }


def _source_file_facts(source_file) -> dict:
    return {
        "id": source_file.id,
        "document_id": source_file.document_id,
        "source_path": source_file.source_path,
        "checksum": source_file.checksum,
        "source_type": source_file.source_type,
        "size_bytes": source_file.size_bytes,
        "metadata_json": dict(source_file.metadata_json or {}),
        "state": source_file.state,
        "created_at": source_file.created_at,
        "updated_at": source_file.updated_at,
    }


async def _seed_existing_document(db_session, knowledge_base, source: Path):
    from app.models import Chunk, ChunkContextText, Document, DocumentVersion, SourceFile, VectorRecord
    from app.services import ingestion

    source.write_text(
        "# Stable source\n\nThe old active document has stable evidence and chunks.\n",
        encoding="utf-8",
    )
    result = await ingestion.ingest_file(
        db_session,
        source,
        knowledge_base_id=knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )
    document = db_session.get(Document, result["document_id"])
    document.title = "Curated old title"
    document.source_type = "curated-markdown"
    document.language = "zh-CN"
    document.tags = ["curated", "stable"]
    document.difficulty = "advanced"
    document.checksum = "old-curated-checksum"
    document.visibility = "restricted"
    document.is_active = True
    db_session.commit()
    db_session.refresh(document)
    source_file = db_session.scalar(
        select(SourceFile).where(
            SourceFile.knowledge_base_id == knowledge_base.id,
            SourceFile.source_path == str(source.resolve()),
        )
    )
    all_chunk_ids = sorted(
        db_session.scalars(select(Chunk.id).where(Chunk.document_id == document.id)).all()
    )
    return {
        "document_id": document.id,
        "metadata": _metadata(document),
        "active_version_ids": sorted(
            db_session.scalars(
                select(DocumentVersion.id).where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.is_active.is_(True),
                )
            ).all()
        ),
        "all_version_ids": sorted(
            db_session.scalars(select(DocumentVersion.id).where(DocumentVersion.document_id == document.id)).all()
        ),
        "active_chunk_ids": sorted(
            db_session.scalars(
                select(Chunk.id).where(Chunk.document_id == document.id, Chunk.state == "active")
            ).all()
        ),
        "all_chunk_ids": all_chunk_ids,
        "context_row_ids": sorted(
            db_session.scalars(
                select(ChunkContextText.id).where(
                    ChunkContextText.chunk_id.in_(all_chunk_ids)
                )
            ).all()
        ),
        "vector_record_ids": sorted(
            db_session.scalars(
                select(VectorRecord.id).where(VectorRecord.chunk_id.in_(all_chunk_ids))
            ).all()
        ),
        "source_file": _source_file_facts(source_file),
        "knowledge_base_chunk_version": knowledge_base.current_chunk_version,
    }


def _install_failure(monkeypatch, stage: str) -> None:
    from app.services import context_graph, ingestion

    if stage == "parser":
        def fail_parser(_path):
            raise RuntimeError("forced parser failure")

        monkeypatch.setattr(ingestion, "parse_document", fail_parser)
        return
    if stage == "embedding":
        class FailingEmbeddingProvider:
            async def embed_texts(self, _texts, text_type="document"):
                assert text_type == "document"
                raise RuntimeError("forced embedding failure")

        monkeypatch.setattr(context_graph, "EmbeddingProvider", FailingEmbeddingProvider)
        return
    if stage == "qdrant":
        class FailingVectorStore:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            async def async_upsert(self, _points) -> None:
                raise RuntimeError("forced qdrant failure")

        monkeypatch.setattr(context_graph, "VectorStore", FailingVectorStore)
        return
    raise AssertionError(f"unknown failure stage: {stage}")


@pytest.mark.asyncio
async def test_ingestion_executor_rejects_out_of_root_worker_path_before_snapshot(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
):
    from app.models import Chunk, DocumentVersion
    from app.services import ingestion
    from app.services.storage import UploadValidationError

    outside = (tmp_path / "tampered-worker-argument.md").resolve()
    outside.write_text("worker must not read these bytes", encoding="utf-8")

    def fail_if_snapshotted(*_args, **_kwargs):
        raise AssertionError("out-of-root source reached snapshot_source_file")

    monkeypatch.setattr(ingestion, "snapshot_source_file", fail_if_snapshotted)
    with pytest.raises(UploadValidationError, match="outside knowledge-base storage"):
        await ingestion.ingest_file(
            db_session,
            outside,
            knowledge_base_id=sample_knowledge_base.id,
            rebuild_graph=False,
            target_version=1,
        )

    assert db_session.scalar(select(Chunk.id)) is None
    assert db_session.scalar(select(DocumentVersion.id)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["parser", "embedding", "qdrant"])
async def test_failed_reparse_restores_document_metadata_version_and_chunks(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
    failure_stage: str,
):
    from app.models import Chunk, ChunkContextText, Document, DocumentVersion, IngestionJob, SourceFile, VectorRecord
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "atomic-metadata.md")
    before = await _seed_existing_document(db_session, sample_knowledge_base, source)
    source.write_text(
        "# Candidate replacement\n\nThis candidate must never become PostgreSQL fact after a failed ingest.\n",
        encoding="utf-8",
    )
    candidate_document, queued_job = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source,
    )
    pending_intent = (queued_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert pending_intent["status"] == "pending"
    assert pending_intent["before_state"]["document_id"] == before["document_id"]
    assert _metadata(candidate_document) == before["metadata"]
    assert pending_intent["candidate_state"]["metadata"]["title"] == before["metadata"]["title"]
    assert pending_intent["candidate_state"]["metadata"]["checksum"] != before["metadata"]["checksum"]
    first_candidate_hash = pending_intent["candidate_state_hash"]
    source.write_text(
        "# Newest replacement\n\nOnly this newest candidate may be attempted; rollback must still restore the last applied facts.\n",
        encoding="utf-8",
    )
    repeated_document, repeated_job = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source,
    )
    repeated_intent = (repeated_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert repeated_job.id == queued_job.id
    assert repeated_document.id == candidate_document.id
    assert _metadata(repeated_document) == before["metadata"]
    assert repeated_intent["before_state"]["metadata"]["title"] == before["metadata"]["title"]
    assert repeated_intent["candidate_state_hash"] != first_candidate_hash
    assert repeated_intent["candidate_revision"] == 2
    assert len(
        db_session.scalars(
            select(IngestionJob).where(
                IngestionJob.knowledge_base_id == sample_knowledge_base.id,
                IngestionJob.source_path == str(source),
            )
        ).all()
    ) == 2  # one completed seed job plus one coalesced pending upload job

    _install_failure(monkeypatch, failure_stage)
    with pytest.raises(RuntimeError, match=f"forced {failure_stage} failure"):
        await ingestion.ingest_file(
            db_session,
            source,
            existing_job_id=queued_job.id,
            knowledge_base_id=sample_knowledge_base.id,
            rebuild_graph=False,
            target_version=1,
        )

    db_session.expire_all()
    restored = db_session.get(Document, before["document_id"])
    failed_job = db_session.get(IngestionJob, queued_job.id)
    assert restored is not None
    assert _metadata(restored) == before["metadata"]
    assert sorted(
        db_session.scalars(
            select(DocumentVersion.id).where(
                DocumentVersion.document_id == restored.id,
                DocumentVersion.is_active.is_(True),
            )
        ).all()
    ) == before["active_version_ids"]
    assert sorted(
        db_session.scalars(select(DocumentVersion.id).where(DocumentVersion.document_id == restored.id)).all()
    ) == before["all_version_ids"]
    assert sorted(
        db_session.scalars(select(Chunk.id).where(Chunk.document_id == restored.id, Chunk.state == "active")).all()
    ) == before["active_chunk_ids"]
    assert sorted(db_session.scalars(select(Chunk.id).where(Chunk.document_id == restored.id)).all()) == before["all_chunk_ids"]
    assert sorted(
        db_session.scalars(
            select(ChunkContextText.id).where(
                ChunkContextText.chunk_id.in_(before["all_chunk_ids"])
            )
        ).all()
    ) == before["context_row_ids"]
    assert sorted(
        db_session.scalars(
            select(VectorRecord.id).where(
                VectorRecord.chunk_id.in_(before["all_chunk_ids"])
            )
        ).all()
    ) == before["vector_record_ids"]
    restored_source_file = db_session.get(SourceFile, before["source_file"]["id"])
    assert _source_file_facts(restored_source_file) == before["source_file"]
    db_session.refresh(sample_knowledge_base)
    assert sample_knowledge_base.current_chunk_version == before["knowledge_base_chunk_version"]
    assert failed_job.status == "failed"
    assert failed_job.document_id == restored.id
    restored_intent = (failed_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert restored_intent["protocol_version"] == ingestion.DOCUMENT_METADATA_INTENT_PROTOCOL_VERSION
    assert restored_intent["status"] == "restored"
    assert restored_intent["restore_verification"]["ok"] is True
    assert restored_intent["restore_verification"]["fact_scope_matches"] is True
    source_snapshot = restored_intent["source_snapshot"]
    assert source_snapshot["protocol_version"] == ingestion.SOURCE_SNAPSHOT_PROTOCOL_VERSION
    assert source_snapshot["checksum"] == repeated_intent["candidate_state"]["metadata"]["checksum"]
    assert source_snapshot["exists"] is True
    assert source_snapshot["checksum_verified"] is True
    assert source_snapshot["disposition"] == "retained_for_audit_or_retry"
    assert failed_job.stats["source_snapshot_recovery"] == source_snapshot
    remaining_pending = [
        job
        for job in db_session.scalars(
            select(IngestionJob).where(
                IngestionJob.knowledge_base_id == sample_knowledge_base.id,
                IngestionJob.source_path == str(source),
            )
        ).all()
        if ((job.stats or {}).get(ingestion.DOCUMENT_METADATA_INTENT_KEY) or {}).get("status") == "pending"
    ]
    assert remaining_pending == []


@pytest.mark.asyncio
async def test_registered_source_change_fails_before_observed_hash_snapshot_commit(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Document, IngestionJob
    from app.services import ingestion
    from app.services.storage import SourceSnapshotError

    source = _kb_source(sample_knowledge_base, "source-slot-race.md")
    before = await _seed_existing_document(db_session, sample_knowledge_base, source)
    source.write_text(
        f"# Registered candidate A\n\nThese bytes are covered by the pending intent for {source}.\n",
        encoding="utf-8",
    )
    _document, queued_job = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source,
    )
    pending_intent = (queued_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    candidate_checksum = pending_intent["candidate_state"]["metadata"]["checksum"]
    candidate_snapshot = ingestion.source_snapshot_path(
        source.name,
        candidate_checksum,
        sample_knowledge_base.name,
    )
    assert candidate_snapshot.exists() is False

    source.write_text(
        f"# External replacement B\n\nThese unregistered bytes replaced {source}.\n",
        encoding="utf-8",
    )
    observed_checksum = ingestion.compute_checksum(source)
    assert observed_checksum != candidate_checksum
    observed_snapshot = ingestion.source_snapshot_path(
        source.name,
        observed_checksum,
        sample_knowledge_base.name,
    )
    assert observed_snapshot.exists() is False

    with pytest.raises(SourceSnapshotError, match="Source bytes changed"):
        await ingestion.ingest_file(
            db_session,
            source,
            knowledge_base_id=sample_knowledge_base.id,
            existing_job_id=queued_job.id,
            rebuild_graph=False,
            target_version=2,
        )

    assert candidate_snapshot.exists() is False
    assert observed_snapshot.exists() is False
    restored_document = db_session.get(Document, before["document_id"])
    failed_job = db_session.get(IngestionJob, queued_job.id)
    assert _metadata(restored_document) == before["metadata"]
    assert failed_job.status == "failed"
    restored_intent = (failed_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert restored_intent["status"] == "restored"
    assert restored_intent["source_snapshot"]["checksum"] == candidate_checksum
    assert restored_intent["source_snapshot"]["exists"] is False
    assert restored_intent["source_snapshot"]["checksum_verified"] is False


@pytest.mark.asyncio
async def test_upload_registration_keeps_existing_document_metadata_applied_until_parse(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Chunk, Document, DocumentVersion
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "upload-only-existing.md")
    before = await _seed_existing_document(db_session, sample_knowledge_base, source)
    source.write_text(
        "# Unparsed candidate\n\nUploading this candidate must not promote its metadata.\n",
        encoding="utf-8",
    )

    registered_document, queued_job = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source,
    )

    db_session.expire_all()
    applied_document = db_session.get(Document, before["document_id"])
    pending_intent = (db_session.get(type(queued_job), queued_job.id).stats or {})[
        ingestion.DOCUMENT_METADATA_INTENT_KEY
    ]
    assert registered_document.id == before["document_id"]
    assert _metadata(applied_document) == before["metadata"]
    assert pending_intent["status"] == "pending"
    assert pending_intent["candidate_state"]["metadata"]["checksum"] != before["metadata"]["checksum"]
    assert sorted(
        db_session.scalars(
            select(DocumentVersion.id).where(
                DocumentVersion.document_id == before["document_id"],
                DocumentVersion.is_active.is_(True),
            )
        ).all()
    ) == before["active_version_ids"]
    assert sorted(
        db_session.scalars(
            select(Chunk.id).where(
                Chunk.document_id == before["document_id"],
                Chunk.state == "active",
            )
        ).all()
    ) == before["active_chunk_ids"]


@pytest.mark.asyncio
async def test_failed_first_parse_removes_candidate_document_but_keeps_failed_job_audit(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
):
    from app.models import Document, IngestionJob
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "first-parse-failure.md")
    source.write_text("# New candidate\n\nThis parse will fail.\n", encoding="utf-8")
    candidate, queued_job = ingestion.register_uploaded_file(db_session, sample_knowledge_base, source)
    candidate_id = candidate.id
    _install_failure(monkeypatch, "parser")

    with pytest.raises(RuntimeError, match="forced parser failure"):
        await ingestion.ingest_file(
            db_session,
            source,
            existing_job_id=queued_job.id,
            knowledge_base_id=sample_knowledge_base.id,
            rebuild_graph=False,
        )

    db_session.expire_all()
    assert db_session.get(Document, candidate_id) is None
    assert db_session.scalar(
        select(Document.id).where(
            Document.knowledge_base_id == sample_knowledge_base.id,
            Document.source_path == str(source),
        )
    ) is None
    failed_job = db_session.get(IngestionJob, queued_job.id)
    assert failed_job.status == "failed"
    assert failed_job.document_id is None
    intent = (failed_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert intent["before_state"]["exists"] is False
    assert intent["status"] == "restored"
    assert intent["restore_verification"]["document_presence_matches"] is True


@pytest.mark.asyncio
async def test_failed_pending_first_parse_allows_unrelated_kb_version_to_advance(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Chunk, Document, IngestionJob, KnowledgeBase
    from app.services import ingestion

    source_a = _kb_source(sample_knowledge_base, "pending-a.md")
    source_a.write_text("# Pending A\n\nA is uploaded first but parsed later.\n", encoding="utf-8")
    candidate_a, queued_a = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source_a,
    )
    candidate_a_id = candidate_a.id
    pending_a = (queued_a.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert pending_a["before_state"]["exists"] is False
    assert pending_a["before_state"]["knowledge_base_chunk_version"] == 0

    source_b = _kb_source(sample_knowledge_base, "successful-b.md")
    source_b.write_text("# Successful B\n\nB advances the knowledge base to version one.\n", encoding="utf-8")
    result_b = await ingestion.ingest_file(
        db_session,
        source_b,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )
    document_b_id = result_b["document_id"]
    db_session.refresh(sample_knowledge_base)
    assert sample_knowledge_base.current_chunk_version == 1

    _install_failure(monkeypatch, "parser")
    with pytest.raises(RuntimeError, match="forced parser failure"):
        await ingestion.ingest_file(
            db_session,
            source_a,
            existing_job_id=queued_a.id,
            knowledge_base_id=sample_knowledge_base.id,
            rebuild_graph=False,
            target_version=1,
        )

    db_session.expire_all()
    assert db_session.get(Document, candidate_a_id) is None
    applied_b = db_session.get(Document, document_b_id)
    assert applied_b is not None
    assert db_session.scalar(
        select(Chunk.id).where(Chunk.document_id == document_b_id, Chunk.state == "active")
    )
    assert db_session.get(KnowledgeBase, sample_knowledge_base.id).current_chunk_version == 1
    failed_a = db_session.get(IngestionJob, queued_a.id)
    restored_intent = (failed_a.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    verification = restored_intent["restore_verification"]
    assert failed_a.status == "failed"
    assert failed_a.document_id is None
    assert restored_intent["status"] == "restored"
    assert verification["ok"] is True
    assert verification["fact_scope_matches"] is True
    assert verification["knowledge_base_version_not_regressed"] is True
    assert verification["knowledge_base_version_advanced"] is True


@pytest.mark.asyncio
async def test_hard_terminated_cancel_restores_pending_intent_before_terminal_state(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import Document, IngestionBatch, IngestionJob
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "sigterm-existing.md")
    before = await _seed_existing_document(db_session, sample_knowledge_base, source)
    source.write_text(
        "# SIGTERM candidate\n\nThis uncommitted promotion must be restored by cancellation finalize.\n",
        encoding="utf-8",
    )
    _registered, queued_job = ingestion.register_uploaded_file(
        db_session,
        sample_knowledge_base,
        source,
    )
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root=str(tmp_path),
        status="cancelling",
        total_files=1,
        stats={
            "phase": "parsing",
            "cancel_requested": True,
            "ingestion_execution_mode": "celery",
            "celery_task_id": "sigterm-task",
            "batch_task_ids": ["sigterm-task"],
        },
    )
    db_session.add(batch)
    db_session.commit()

    promoted_document, staged_job, pending_intent = ingestion._stage_document_metadata_for_ingestion(
        db_session,
        knowledge_base=sample_knowledge_base,
        path=source,
        checksum=ingestion.compute_checksum(source),
        source_type=ingestion.source_type_from_path(source),
        trigger_source="upload",
        batch_id=batch.id,
        existing_job_id=queued_job.id,
    )
    assert staged_job.batch_id == batch.id
    assert promoted_document.checksum == pending_intent["candidate_state"]["metadata"]["checksum"]
    # Simulate SIGTERM closing the worker connection before ingest_file can run
    # its exception handler: only the uncommitted main transaction is rolled back.
    db_session.rollback()

    monkeypatch.setattr(
        ingestion,
        "inspect_celery_batch_tasks",
        lambda task_ids: {
            "attempted": True,
            "ok": True,
            "task_ids": task_ids,
            "active_task_ids": [],
            "reserved_task_ids": [],
            "scheduled_task_ids": [],
        },
    )
    released_batch = db_session.get(IngestionBatch, batch.id)
    ingestion.finalize_cancelling_batch_if_released(db_session, released_batch)

    db_session.expire_all()
    restored_document = db_session.get(Document, before["document_id"])
    restored_batch = db_session.get(IngestionBatch, batch.id)
    restored_job = db_session.get(IngestionJob, queued_job.id)
    restored_intent = (restored_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert _metadata(restored_document) == before["metadata"]
    assert restored_batch.status == "cancelled"
    assert restored_batch.stats["metadata_intent_recovery"]["ok"] is True
    assert restored_batch.stats["metadata_intent_recovery"]["restored_job_ids"] == [queued_job.id]
    assert restored_job.status == "cancelled"
    assert restored_intent["status"] == "restored"
    assert restored_intent["restore_verification"]["ok"] is True
    source_snapshot = restored_intent["source_snapshot"]
    assert source_snapshot["protocol_version"] == ingestion.SOURCE_SNAPSHOT_PROTOCOL_VERSION
    assert source_snapshot["checksum"] == pending_intent["candidate_state"]["metadata"]["checksum"]
    assert source_snapshot["exists"] is False
    assert source_snapshot["checksum_verified"] is False
    assert source_snapshot["disposition"] == "retained_for_audit_or_retry"


def test_released_cancel_with_corrupt_pending_intent_fails_closed(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
):
    from app.models import IngestionBatch, IngestionJob
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "corrupt-cancel.md")
    source.write_text("# Corrupt pending intent\n", encoding="utf-8")
    _candidate, queued_job = ingestion.register_uploaded_file(db_session, sample_knowledge_base, source)
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root=str(tmp_path),
        status="cancelling",
        total_files=1,
        stats={
            "cancel_requested": True,
            "ingestion_execution_mode": "celery",
            "celery_task_id": "corrupt-task",
            "batch_task_ids": ["corrupt-task"],
        },
    )
    db_session.add(batch)
    db_session.flush()
    queued_job.batch_id = batch.id
    intent = dict((queued_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY])
    intent["pending_payload_hash"] = "tampered"
    queued_job.stats = {**(queued_job.stats or {}), ingestion.DOCUMENT_METADATA_INTENT_KEY: intent}
    db_session.commit()
    monkeypatch.setattr(
        ingestion,
        "inspect_celery_batch_tasks",
        lambda task_ids: {
            "attempted": True,
            "ok": True,
            "task_ids": task_ids,
            "active_task_ids": [],
            "reserved_task_ids": [],
            "scheduled_task_ids": [],
        },
    )

    ingestion.finalize_cancelling_batch_if_released(db_session, batch)

    db_session.expire_all()
    failed_batch = db_session.get(IngestionBatch, batch.id)
    failed_job = db_session.get(IngestionJob, queued_job.id)
    assert failed_batch.status == "cancel_failed"
    assert failed_batch.stats["manual_review_required"] is True
    assert failed_batch.stats["metadata_intent_recovery"]["ok"] is False
    assert failed_job.status == "failed"
    assert failed_job.stats["manual_review_required"] is True
    assert failed_job.stats["phase"] == "metadata_release_recovery_failed"


def test_released_cancel_refuses_to_delete_candidate_with_committed_facts(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
):
    from app.models import Document, IngestionBatch, IngestionJob, SourceFile
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "candidate-with-facts.md")
    source.write_text("# Candidate with independently committed facts\n", encoding="utf-8")
    candidate, queued_job = ingestion.register_uploaded_file(db_session, sample_knowledge_base, source)
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root=str(tmp_path),
        status="cancelling",
        total_files=1,
        stats={
            "cancel_requested": True,
            "ingestion_execution_mode": "celery",
            "celery_task_id": "committed-facts-task",
            "batch_task_ids": ["committed-facts-task"],
        },
    )
    db_session.add(batch)
    db_session.flush()
    queued_job.batch_id = batch.id
    committed_source_file = SourceFile(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=candidate.id,
        source_path=str(source),
        checksum=ingestion.compute_checksum(source),
        source_type=ingestion.source_type_from_path(source),
        size_bytes=source.stat().st_size,
        metadata_json={"origin": "independent_commit"},
        state="active",
    )
    db_session.add(committed_source_file)
    db_session.commit()
    monkeypatch.setattr(
        ingestion,
        "inspect_celery_batch_tasks",
        lambda task_ids: {
            "attempted": True,
            "ok": True,
            "task_ids": task_ids,
            "active_task_ids": [],
            "reserved_task_ids": [],
            "scheduled_task_ids": [],
        },
    )

    ingestion.finalize_cancelling_batch_if_released(db_session, batch)

    db_session.expire_all()
    failed_batch = db_session.get(IngestionBatch, batch.id)
    failed_job = db_session.get(IngestionJob, queued_job.id)
    retained_document = db_session.get(Document, candidate.id)
    retained_source_file = db_session.get(SourceFile, committed_source_file.id)
    failed_intent = (failed_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert failed_batch.status == "cancel_failed"
    assert failed_batch.stats["manual_review_required"] is True
    assert failed_job.status == "failed"
    assert failed_job.stats["manual_review_required"] is True
    assert failed_intent["status"] == "restore_failed"
    assert retained_document is not None
    assert retained_source_file is not None
    assert retained_source_file.document_id == retained_document.id


@pytest.mark.parametrize("worker_released", [False, True])
def test_interrupted_worker_recovers_pending_intent_only_after_release_proof(
    monkeypatch,
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    worker_released: bool,
):
    from app.models import Document, IngestionBatch, IngestionJob
    from app.services import ingestion

    source = _kb_source(
        sample_knowledge_base,
        f"interrupted-{worker_released}.md",
    )
    source.write_text("# Interrupted candidate\n", encoding="utf-8")
    candidate, queued_job = ingestion.register_uploaded_file(db_session, sample_knowledge_base, source)
    candidate_id = candidate.id
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        trigger_source="upload",
        source_root=str(tmp_path),
        status="parsing",
        total_files=1,
        stats={
            "phase": "parsing",
            "ingestion_execution_mode": "celery",
            "celery_task_id": "restart-task",
            "batch_task_ids": ["restart-task"],
        },
    )
    db_session.add(batch)
    db_session.flush()
    queued_job.batch_id = batch.id
    queued_job.status = "processing"
    db_session.commit()
    monkeypatch.setattr(
        ingestion,
        "inspect_celery_batch_tasks",
        lambda task_ids: {
            "attempted": True,
            "ok": True,
            "task_ids": task_ids,
            "active_task_ids": [] if worker_released else task_ids,
            "reserved_task_ids": [],
            "scheduled_task_ids": [],
        },
    )
    monkeypatch.setattr(
        ingestion,
        "inspect_celery_batch_task_results",
        lambda task_ids: {
            "attempted": True,
            "ok": True,
            "task_ids": task_ids,
            "states": {
                task_id: ("FAILURE" if worker_released else "PENDING")
                for task_id in task_ids
            },
            "terminal_task_ids": task_ids if worker_released else [],
            "errors": [],
        },
    )

    ingestion.finalize_interrupted_batches()

    db_session.expire_all()
    observed_batch = db_session.get(IngestionBatch, batch.id)
    observed_job = db_session.get(IngestionJob, queued_job.id)
    observed_intent = (observed_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    if worker_released:
        assert observed_batch.status == "failed"
        assert observed_batch.stats["metadata_intent_recovery"]["ok"] is True
        assert observed_job.status == "failed"
        assert observed_intent["status"] == "restored"
        assert db_session.get(Document, candidate_id) is None
    else:
        assert observed_batch.status == "parsing"
        assert observed_batch.stats["interrupted_recovery_deferred"]["release_proof_required"] is True
        assert observed_job.status == "processing"
        assert observed_intent["status"] == "pending"
        assert db_session.get(Document, candidate_id) is not None


@pytest.mark.asyncio
async def test_successful_ingest_reuses_upload_intent_and_marks_it_applied(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    fake_model_stack,
):
    from app.models import IngestionJob
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "successful-intent.md")
    source.write_text("# Successful intent\n\nGrounded content.\n", encoding="utf-8")
    document, queued_job = ingestion.register_uploaded_file(db_session, sample_knowledge_base, source)

    result = await ingestion.ingest_file(
        db_session,
        source,
        knowledge_base_id=sample_knowledge_base.id,
        rebuild_graph=False,
        target_version=1,
    )

    db_session.expire_all()
    applied_job = db_session.get(IngestionJob, queued_job.id)
    all_jobs = db_session.scalars(
        select(IngestionJob).where(
            IngestionJob.knowledge_base_id == sample_knowledge_base.id,
            IngestionJob.source_path == str(source),
        )
    ).all()
    assert result["document_id"] == document.id
    assert len(all_jobs) == 1
    assert applied_job.status == "completed"
    intent = (applied_job.stats or {})[ingestion.DOCUMENT_METADATA_INTENT_KEY]
    assert intent["status"] == "applied"
    assert intent["apply_verification"]["ok"] is True
    assert intent["apply_verification"]["active_chunk_ids"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_before_metadata_field",
        "extra_before_state_field",
        "extra_candidate_state_field",
        "missing_candidate_metadata_field",
        "extra_intent_envelope_field",
    ],
)
def test_document_metadata_intent_schema_rejects_rehashed_malformed_packets(
    tmp_path: Path,
    db_session,
    sample_knowledge_base,
    mutation: str,
):
    from app.models import Document
    from app.services import ingestion

    source = _kb_source(sample_knowledge_base, "strict-metadata-intent.md")
    source.write_text("# Strict schema\n", encoding="utf-8")
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Applied title",
        source_path=str(source),
        source_type="markdown",
        language="zh-CN",
        tags=["applied"],
        difficulty="advanced",
        checksum="applied-checksum",
        visibility="private",
        is_active=True,
    )
    db_session.add(document)
    db_session.commit()
    intent = ingestion._build_document_metadata_intent(
        db_session,
        knowledge_base=sample_knowledge_base,
        document=document,
        path=source,
        checksum="candidate-checksum",
        source_type="markdown",
    )
    intent["document_id"] = document.id

    if mutation == "missing_before_metadata_field":
        del intent["before_state"]["metadata"]["title"]
        intent["before_state_hash"] = ingestion._canonical_payload_hash(intent["before_state"])
    elif mutation == "extra_before_state_field":
        intent["before_state"]["unexpected"] = True
        intent["before_state_hash"] = ingestion._canonical_payload_hash(intent["before_state"])
    elif mutation == "extra_candidate_state_field":
        intent["candidate_state"]["unexpected"] = True
        intent["candidate_state_hash"] = ingestion._canonical_payload_hash(intent["candidate_state"])
    elif mutation == "missing_candidate_metadata_field":
        del intent["candidate_state"]["metadata"]["title"]
        intent["candidate_state_hash"] = ingestion._canonical_payload_hash(intent["candidate_state"])
    elif mutation == "extra_intent_envelope_field":
        intent["unexpected"] = True
    else:  # pragma: no cover
        raise AssertionError(mutation)
    intent = ingestion._refresh_pending_intent_payload_hash(intent)

    with pytest.raises(ingestion.DocumentMetadataRestoreError):
        ingestion._validate_document_metadata_intent(
            intent,
            knowledge_base_id=sample_knowledge_base.id,
            path=source,
            checksum="candidate-checksum",
            source_type="markdown",
        )

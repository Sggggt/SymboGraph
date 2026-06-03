from __future__ import annotations

import pytest


def test_request_batch_cancel_requires_matching_course_scope(db_session, sample_course):
    from app.models import Course, IngestionBatch
    from app.services.ingestion import request_batch_cancel

    other_course = Course(name="Other Course", description="tests", source_root="other")
    db_session.add(other_course)
    db_session.flush()
    active_batch = IngestionBatch(
        course_id=sample_course.id,
        source_root="unit",
        trigger_source="upload",
        status="embedding",
        total_files=1,
    )
    terminal_batch = IngestionBatch(
        course_id=sample_course.id,
        source_root="unit",
        trigger_source="upload",
        status="completed",
        total_files=1,
    )
    db_session.add_all([active_batch, terminal_batch])
    db_session.commit()

    with pytest.raises(PermissionError):
        request_batch_cancel(db_session, active_batch.id, other_course.id)
    with pytest.raises(PermissionError):
        request_batch_cancel(db_session, active_batch.id, "")
    with pytest.raises(PermissionError):
        request_batch_cancel(db_session, terminal_batch.id, other_course.id)

    result = request_batch_cancel(db_session, active_batch.id, sample_course.id)
    assert result is not None
    assert result["state"] == "cancel_requested"


@pytest.mark.asyncio
async def test_cancelled_batch_restores_previous_version_and_deletes_new_vectors(db_session, sample_course, monkeypatch):
    from datetime import datetime

    from app.models import Chunk, Document, DocumentVersion, IngestionBatch, IngestionJob
    from app.services import ingestion
    from app.services.ingestion import compensate_cancelled_batch, request_batch_cancel

    class FakeVectorStore:
        deleted: list[str] = []
        upserted: list[dict] = []

        def __init__(self, *args, **kwargs):
            pass

        async def async_delete(self, ids):
            self.deleted.extend(ids)

        async def async_upsert(self, points):
            self.upserted.extend(points)

    monkeypatch.setattr(ingestion, "VectorStore", FakeVectorStore)

    document = Document(
        course_id=sample_course.id,
        title="Cancelable",
        source_path="cancel.md",
        source_type="markdown",
        checksum="old",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    old_version = DocumentVersion(document_id=document.id, version=1, checksum="old", storage_path="cancel.md", is_active=True)
    db_session.add(old_version)
    db_session.flush()
    old_chunk = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=old_version.id,
        content="old",
        snippet="old",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        is_active=True,
    )
    db_session.add(old_chunk)
    db_session.flush()
    batch = IngestionBatch(
        course_id=sample_course.id,
        source_root="unit",
        trigger_source="upload",
        status="embedding",
        started_at=datetime.utcnow(),
        total_files=1,
    )
    db_session.add(batch)
    db_session.flush()
    job = IngestionJob(
        course_id=sample_course.id,
        batch_id=batch.id,
        document_id=document.id,
        source_path=document.source_path,
        trigger_source="upload",
        status="embedding",
        stats={"pre_batch_document_id": document.id, "pre_batch_active_version_id": old_version.id, "pre_batch_active_version": 1},
    )
    db_session.add(job)
    db_session.flush()
    new_version = DocumentVersion(document_id=document.id, version=2, checksum="new", storage_path="cancel.md", is_active=True)
    old_version.is_active = False
    db_session.add(new_version)
    db_session.flush()
    new_chunk = Chunk(
        course_id=sample_course.id,
        document_id=document.id,
        document_version_id=new_version.id,
        content="new",
        snippet="new",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        is_active=True,
    )
    old_chunk.is_active = False
    db_session.add(new_chunk)
    db_session.commit()

    request_batch_cancel(db_session, batch.id, sample_course.id)
    stats = await compensate_cancelled_batch(db_session, batch.id)

    db_session.refresh(batch)
    db_session.refresh(old_version)
    db_session.refresh(new_version)
    db_session.refresh(old_chunk)
    db_session.refresh(new_chunk)
    assert batch.status == "cancelled"
    assert old_version.is_active is True
    assert old_chunk.is_active is True
    assert new_version.is_active is False
    assert new_chunk.is_active is False
    assert new_chunk.id in FakeVectorStore.deleted
    assert stats["restored_document_versions"] == 1


@pytest.mark.asyncio
async def test_run_graph_rebuild_incremental_cancel_does_not_fallback_to_full(db_session, sample_course, monkeypatch):
    from datetime import datetime, timedelta

    from app.models import Document, IngestionBatch
    from app.services import concept_graph, ingestion
    from app.services.cancellation import IngestionCancelled
    from app.services.ingestion import run_graph_rebuild

    class FakeVectorStore:
        def __init__(self, *args, **kwargs):
            pass

        async def async_delete(self, ids):
            pass

        async def async_upsert(self, points):
            pass

    last_batch = IngestionBatch(
        course_id=sample_course.id,
        source_root="graph",
        trigger_source="rebuild_graph",
        status="completed",
        started_at=datetime.utcnow() - timedelta(hours=2),
        completed_at=datetime.utcnow() - timedelta(hours=1),
    )
    current_batch = IngestionBatch(
        course_id=sample_course.id,
        source_root="graph",
        trigger_source="rebuild_graph",
        status="queued",
        total_files=0,
    )
    document = Document(
        course_id=sample_course.id,
        title="Changed",
        source_path="changed.md",
        source_type="markdown",
        checksum="changed",
        tags=[],
        is_active=True,
    )
    db_session.add_all([last_batch, current_batch, document])
    db_session.commit()

    async def fake_incremental_update(session, course_id, changed_document_ids, batch_id):
        assert changed_document_ids == [document.id]
        raise IngestionCancelled("unit cancel")

    async def forbidden_full_rebuild(*args, **kwargs):
        raise AssertionError("cancelled incremental rebuild must not fall back to full rebuild")

    monkeypatch.setattr(ingestion, "acquire_course_graph_lock", lambda session, course_id: True)
    monkeypatch.setattr(ingestion, "release_course_graph_lock", lambda session, course_id: None)
    monkeypatch.setattr(ingestion, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(ingestion, "rebuild_course_graph", forbidden_full_rebuild)
    monkeypatch.setattr(concept_graph, "incremental_update_course_graph", fake_incremental_update)
    monkeypatch.setattr(concept_graph, "_restore_course_graph_from_backup", lambda session, course_id: None)

    result = await run_graph_rebuild(current_batch.id, sample_course.id, mode="incremental")

    db_session.expire_all()
    refreshed = db_session.get(IngestionBatch, current_batch.id)
    assert result["state"] == "cancelled"
    assert refreshed.status == "cancelled"


@pytest.mark.asyncio
async def test_uploaded_parse_marks_batch_extracting_graph_before_rebuild(db_session, sample_course, monkeypatch):
    from app.models import IngestionBatch
    from app.services import ingestion
    from app.services.ingestion import create_uploaded_files_batch, run_uploaded_files_ingestion

    from app.core.config import get_settings

    storage_root = get_settings().course_paths_for_name(sample_course.name)["storage_root"]
    source_path = storage_root / "note.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Note\n\nFresh content", encoding="utf-8")

    batch = create_uploaded_files_batch(db_session, sample_course.id, [source_path], force=False)
    observed_statuses: list[str] = []

    async def fake_ingest_file(*args, **kwargs):
        return {"status": "completed", "source_type": "markdown", "stats": {}}

    async def fake_rebuild_course_graph(session, course_id):
        current = session.get(IngestionBatch, batch.id)
        observed_statuses.append(current.status if current else "missing")
        return {
            "graph_rebuilt": True,
            "graph_nodes": 1,
            "graph_edges": 0,
            "concepts": 1,
            "relations": 0,
            "graph_extraction_provider": "unit-test",
        }

    monkeypatch.setattr(ingestion, "ingest_file", fake_ingest_file)
    monkeypatch.setattr(ingestion, "rebuild_course_graph", fake_rebuild_course_graph)

    result = await run_uploaded_files_ingestion(batch.id, [str(source_path)], force=False, rebuild_graph_mode="full")

    assert observed_statuses == ["extracting_graph"]
    assert result["state"] == "completed"


@pytest.mark.asyncio
async def test_uploaded_parse_marks_graph_failure_terminal(db_session, sample_course, monkeypatch):
    from app.core.config import get_settings
    from app.models import IngestionBatch
    from app.services import ingestion
    from app.services.ingestion import create_uploaded_files_batch, run_uploaded_files_ingestion

    storage_root = get_settings().course_paths_for_name(sample_course.name)["storage_root"]
    source_path = storage_root / "graph-failure.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Note\n\nFresh content", encoding="utf-8")

    batch = create_uploaded_files_batch(db_session, sample_course.id, [source_path], force=False)

    async def fake_ingest_file(*args, **kwargs):
        return {"status": "completed", "source_type": "markdown", "stats": {}}

    async def fake_rebuild_course_graph(session, course_id):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ingestion, "ingest_file", fake_ingest_file)
    monkeypatch.setattr(ingestion, "rebuild_course_graph", fake_rebuild_course_graph)

    result = await run_uploaded_files_ingestion(batch.id, [str(source_path)], force=False, rebuild_graph_mode="full")
    db_session.expire_all()
    refreshed = db_session.get(IngestionBatch, batch.id)

    assert result["state"] == "partial_failed"
    assert refreshed.status == "partial_failed"
    assert refreshed.completed_at is not None
    assert "图谱生成失败" in refreshed.last_error


@pytest.mark.asyncio
async def test_ingest_short_markdown_keeps_parent_and_child_chunks(db_session, sample_course, monkeypatch):
    from app.core.config import get_settings
    from app.models import Chunk
    from app.services import ingestion
    from app.services.embeddings import EmbeddingCallResult
    from app.services.ingestion import ingest_file

    class FakeEmbeddingProvider:
        settings = get_settings()

        async def embed_texts_with_meta(self, texts, text_type="document"):
            return EmbeddingCallResult(
                vectors=[[1.0, 0.0] for _ in texts],
                provider="unit-test",
                external_called=False,
                fallback_reason=None,
            )

    class FakeChatProvider:
        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            return fallback

    class FakeVectorStore:
        def __init__(self, *args, **kwargs):
            self.points = {}

        def upsert(self, points):
            self.points.update({point["id"]: point for point in points})

        def get_points(self, ids):
            return [self.points[item] for item in ids if item in self.points]

        def health_check(self, course_id, active_chunk_ids):
            return {"ok": True, "missing": [], "stale": []}

        def delete(self, ids):
            for item in ids:
                self.points.pop(item, None)

        async def async_upsert(self, points):
            self.upsert(points)

        async def async_delete(self, ids):
            self.delete(ids)

    monkeypatch.setattr(ingestion, "EmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr(ingestion, "ChatProvider", FakeChatProvider)
    monkeypatch.setattr(ingestion, "VectorStore", FakeVectorStore)

    storage_root = get_settings().course_paths_for_name(sample_course.name)["storage_root"]
    source_path = storage_root / "short-centrality.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "# Centrality smoke test\n\n"
        "Degree centrality counts incident edges in a graph. "
        "It is a local network-analysis measure used to compare node prominence.",
        encoding="utf-8",
    )

    result = await ingest_file(db_session, source_path, course_id=sample_course.id, rebuild_graph=False, force=True)

    chunks = db_session.query(Chunk).filter(Chunk.document_id == result["document_id"], Chunk.is_active.is_(True)).all()
    assert any((chunk.metadata_json or {}).get("is_parent") for chunk in chunks)
    assert any(not (chunk.metadata_json or {}).get("is_parent") for chunk in chunks)
    assert result["stats"]["chunks"] >= 2

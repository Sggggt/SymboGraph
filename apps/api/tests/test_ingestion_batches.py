from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest


def _seed_active_chunk(
    db_session,
    sample_knowledge_base,
    document,
    version,
    *,
    content: str,
    snippet: str | None = None,
    chunk_version: int | None = None,
    state: str = "active",
    metadata: dict | None = None,
):
    from app.models import ActiveChunk, ChunkCandidate, ChunkDecision, EvidenceAtom, EvidenceGraphState, PolicyState, QualityDecision
    from app.services.evidence_graph import stable_hash

    snippet = snippet or content[:120]
    metadata = dict(metadata or {})
    atom_index = db_session.query(EvidenceAtom).filter(EvidenceAtom.document_version_id == version.id).count()
    policy_state = PolicyState(
        knowledge_base_id=sample_knowledge_base.id,
        profile_objective_hash="test-profile",
        posterior_json={},
        constraints_json={},
        exploration_json={},
        reward_summary_json={},
        state_hash=stable_hash({"policy": document.id, "content": content, "atom_index": atom_index}),
    )
    db_session.add(policy_state)
    db_session.flush()
    atom = EvidenceAtom(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=document.id,
        document_version_id=version.id,
        atom_index=atom_index,
        atom_type="paragraph",
        text=content,
        text_hash=stable_hash({"text": content}),
        source_span_json={"spans": [{"start": 0, "end": len(content)}]},
        layout_json={},
        metadata_json={},
        state="active" if state == "active" else "inactive",
    )
    db_session.add(atom)
    db_session.flush()
    graph_state = EvidenceGraphState(
        knowledge_base_id=sample_knowledge_base.id,
        scope_type="global",
        state_hash=stable_hash({"graph": atom.id}),
        atom_scope_hash=stable_hash({"atoms": [atom.id]}),
        active_document_version_ids=[version.id],
        active_atom_ids=[atom.id],
        policy_state_id=policy_state.id,
        stats_json={},
        diagnostics_json={},
        state="active",
    )
    db_session.add(graph_state)
    db_session.flush()
    candidate = ChunkCandidate(
        graph_state_id=graph_state.id,
        generator_name="test_fixture",
        generator_version="test_v1",
        atom_ids_json=[atom.id],
        source_span_union_json={"spans": [{"start": 0, "end": len(content)}]},
        token_count=16,
        graph_features_json={},
        cost_json={},
        diagnostics_json={},
    )
    db_session.add(candidate)
    db_session.flush()
    quality = QualityDecision(
        candidate_id=candidate.id,
        policy_state_id=policy_state.id,
        decision_action="answer_candidate",
        gate_passed=True,
        confidence=1.0,
        diagnostics_json={},
        reward_features_json={},
        feedback_json={},
    )
    db_session.add(quality)
    db_session.flush()
    decision = ChunkDecision(
        knowledge_base_id=sample_knowledge_base.id,
        graph_state_id=graph_state.id,
        candidate_id=candidate.id,
        quality_decision_id=quality.id,
        policy_state_id=policy_state.id,
        action="activate",
    )
    db_session.add(decision)
    db_session.flush()
    chunk = ActiveChunk(
        knowledge_base_id=sample_knowledge_base.id,
        chunk_decision_id=decision.id,
        document_version_scope_hash=stable_hash({"version": version.id}),
        graph_state_hash=graph_state.state_hash,
        atom_ids_json=[atom.id],
        text=content,
        source_span_union_json={"spans": [{"evidence_atom_id": atom.id, "start": 0, "end": len(content)}]},
        boundary_policy_version="test_v1",
        quality_decision_id=quality.id,
        policy_state_id=policy_state.id,
        community_ids_json=[],
        metadata_json={
            "document_id": document.id,
            "document_version_id": version.id,
            "chunk_version": chunk_version if chunk_version is not None else version.version,
            "source_type": document.source_type,
            "snippet": snippet,
            "content_kind": "markdown",
            "is_parent": False,
            **metadata,
        },
        state=state,
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def test_selected_parse_aligns_to_current_chunk_version_without_bump(db_session, sample_knowledge_base, tmp_path):
    from app.models import Document, DocumentVersion
    from app.services.ingestion import create_or_update_document

    sample_knowledge_base.current_chunk_version = 3
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Existing",
        source_path=str(tmp_path / "existing.md"),
        source_type="markdown",
        checksum="old",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(document_id=document.id, version=1, checksum="old", storage_path=document.source_path, is_active=True)
    db_session.add(version)
    db_session.flush()
    _seed_active_chunk(db_session, sample_knowledge_base, document, version, content="old", snippet="old", chunk_version=1)
    db_session.commit()

    path = tmp_path / "existing.md"
    path.write_text("new", encoding="utf-8")
    _document, version_number, stale_chunk_ids = create_or_update_document(
        db_session,
        sample_knowledge_base,
        path,
        "Existing",
        "markdown",
        "new",
        force=True,
    )

    assert version_number == 3
    assert sample_knowledge_base.current_chunk_version == 3
    assert stale_chunk_ids


@pytest.mark.asyncio
async def test_chunk_knowledge_generation_timeout_is_nonblocking(monkeypatch):
    from app.services import ingestion

    class SlowChat:
        async def classify_json(self, **_kwargs):
            await asyncio.sleep(1)
            return {"results": [{"summary": "should not be used", "keywords": ["x"]}]}

    chunk = SimpleNamespace(content="centrality and modularity", summary=None, keywords=[])
    monkeypatch.setattr(
        ingestion,
        "get_settings",
        lambda: SimpleNamespace(model_request_timeout_seconds=0.01),
    )

    await ingestion._generate_chunk_knowledge([chunk], SlowChat(), batch_id=None)

    assert chunk.summary is None
    assert chunk.keywords == []


def test_empty_library_first_parse_uses_v1_and_full_reparse_uses_explicit_target(db_session, sample_knowledge_base, tmp_path):
    from app.services.ingestion import create_or_update_document

    first_path = tmp_path / "first.md"
    first_path.write_text("first", encoding="utf-8")
    _document, version_number, stale_chunk_ids = create_or_update_document(
        db_session,
        sample_knowledge_base,
        first_path,
        "First",
        "markdown",
        "first",
        force=True,
    )
    assert version_number == 1
    assert stale_chunk_ids == []

    _document, version_number, _stale_chunk_ids = create_or_update_document(
        db_session,
        sample_knowledge_base,
        first_path,
        "First",
        "markdown",
        "first-new",
        force=True,
        target_version=2,
    )
    assert version_number == 2


def test_chunk_dedup_ignores_active_chunks_below_target_version(db_session, sample_knowledge_base):
    from app.models import Document, DocumentVersion
    from app.services.ingestion import (
        active_chunk_hashes_for_knowledge_base,
        chunk_content_hash,
        deactivate_superseded_duplicate_documents_by_chunks,
    )

    old_document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Current",
        source_path="storage_temp/legacy.md",
        source_type="markdown",
        checksum="old",
        tags=[],
        is_active=True,
    )
    current_document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Current",
        source_path="storage/current.md",
        source_type="markdown",
        checksum="current",
        tags=[],
        is_active=True,
    )
    db_session.add_all([old_document, current_document])
    db_session.flush()
    old_version = DocumentVersion(document_id=old_document.id, version=1, checksum="old", storage_path="legacy", is_active=True)
    current_version = DocumentVersion(document_id=current_document.id, version=2, checksum="current", storage_path="current", is_active=True)
    db_session.add_all([old_version, current_version])
    db_session.flush()
    duplicated_content = "Bayesian posterior is proportional to likelihood times prior."
    _seed_active_chunk(
        db_session,
        sample_knowledge_base,
        old_document,
        old_version,
        content=duplicated_content,
        snippet=duplicated_content,
        chunk_version=1,
        metadata={"is_parent": False},
    )
    _seed_active_chunk(
        db_session,
        sample_knowledge_base,
        current_document,
        current_version,
        content="A conjugate prior keeps the posterior in the same family.",
        snippet="A conjugate prior keeps the posterior in the same family.",
        chunk_version=2,
        metadata={"is_parent": False},
    )
    db_session.commit()

    target_hashes = active_chunk_hashes_for_knowledge_base(db_session, sample_knowledge_base.id, minimum_chunk_version=2)
    all_hashes = active_chunk_hashes_for_knowledge_base(db_session, sample_knowledge_base.id)

    assert chunk_content_hash(duplicated_content) in all_hashes
    assert chunk_content_hash(duplicated_content) not in target_hashes
    assert chunk_content_hash("A conjugate prior keeps the posterior in the same family.") in target_hashes

    stale_chunk_ids = deactivate_superseded_duplicate_documents_by_chunks(
        db_session,
        knowledge_base_id=sample_knowledge_base.id,
        canonical_document_id=current_document.id,
        title=current_document.title,
        replacement_chunk_hashes={chunk_content_hash(duplicated_content)},
        target_version=2,
    )
    db_session.commit()
    db_session.refresh(old_document)
    db_session.refresh(old_version)
    db_session.refresh(current_document)

    assert stale_chunk_ids
    assert old_document.is_active is False
    assert old_version.is_active is False
    assert current_document.is_active is True


def test_request_batch_cancel_requires_matching_knowledge_base_scope(db_session, sample_knowledge_base):
    from app.models import KnowledgeBase, IngestionBatch
    from app.services.ingestion import request_batch_cancel

    other_course = KnowledgeBase(name="Other KnowledgeBase", description="tests", source_root="other")
    db_session.add(other_course)
    db_session.flush()
    active_batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="unit",
        trigger_source="upload",
        status="embedding",
        total_files=1,
    )
    terminal_batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
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

    result = request_batch_cancel(db_session, active_batch.id, sample_knowledge_base.id)
    assert result is not None
    assert result["state"] == "cancel_requested"


@pytest.mark.asyncio
async def test_control_plane_cancel_finishes_unclaimed_graph_batch(db_session, sample_knowledge_base, monkeypatch):
    from app.models import IngestionBatch
    from app.services.ingestion import active_batch_for_knowledge_base, request_batch_cancel_control

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="graph_rebuild",
        trigger_source="rebuild_graph",
        status="extracting_graph",
        total_files=0,
        stats={"phase": "graph", "parse_committed": True},
    )
    db_session.add(batch)
    db_session.commit()

    result = await request_batch_cancel_control(db_session, batch.id, sample_knowledge_base.id)

    db_session.refresh(batch)
    assert result is not None
    assert result["state"] == "cancelled"
    assert batch.status == "cancelled"
    assert batch.completed_at is not None
    assert batch.stats["evidence_graph_retained"] is True
    assert active_batch_for_knowledge_base(db_session, sample_knowledge_base.id) is None


@pytest.mark.asyncio
async def test_control_plane_cancel_leaves_live_worker_batch_requested(db_session, sample_knowledge_base):
    from datetime import datetime

    from app.models import IngestionBatch
    from app.services.ingestion import active_batch_for_knowledge_base, request_batch_cancel_control

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="graph_rebuild",
        trigger_source="rebuild_graph",
        status="extracting_graph",
        started_at=datetime.utcnow(),
        worker_id="worker-1",
        heartbeat_at=datetime.utcnow(),
        stats={"phase": "graph", "parse_committed": True},
    )
    db_session.add(batch)
    db_session.commit()

    result = await request_batch_cancel_control(db_session, batch.id, sample_knowledge_base.id)

    db_session.refresh(batch)
    assert result is not None
    assert result["state"] == "cancel_requested"
    assert batch.status == "cancel_requested"
    assert active_batch_for_knowledge_base(db_session, sample_knowledge_base.id).id == batch.id


def test_set_job_state_does_not_overwrite_cancel_request(db_session, sample_knowledge_base):
    from app.models import IngestionBatch, IngestionJob
    from app.services.cancellation import IngestionCancelled
    from app.services.ingestion import set_job_state

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="unit",
        trigger_source="upload",
        status="cancel_requested",
        stats={"cancel_requested": True, "cancellation_status": "requested"},
    )
    job = IngestionJob(
        knowledge_base_id=sample_knowledge_base.id,
        batch_id=batch.id,
        source_path="cancel.md",
        trigger_source="upload",
        status="queued",
    )
    db_session.add_all([batch, job])
    db_session.commit()

    with pytest.raises(IngestionCancelled):
        set_job_state(db_session, job, "parsing", batch_id=batch.id)

    db_session.refresh(batch)
    db_session.refresh(job)
    assert batch.status == "cancel_requested"
    assert batch.stats["cancel_requested"] is True
    assert job.status == "queued"


@pytest.mark.asyncio
async def test_cancelled_batch_restores_previous_version_and_deletes_new_vectors(db_session, sample_knowledge_base, monkeypatch):
    from datetime import datetime

    from app.models import Document, DocumentVersion, IngestionBatch, IngestionJob
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
        knowledge_base_id=sample_knowledge_base.id,
        title="Cancelable",
        source_path="cancel.md",
        source_type="markdown",
        checksum="old",
        tags=[],
        is_active=True,
    )
    db_session.add(document)
    sample_knowledge_base.current_chunk_version = 2
    db_session.flush()
    old_version = DocumentVersion(document_id=document.id, version=1, checksum="old", storage_path="cancel.md", is_active=True)
    db_session.add(old_version)
    db_session.flush()
    old_chunk = _seed_active_chunk(db_session, sample_knowledge_base, document, old_version, content="old", snippet="old", chunk_version=1)
    db_session.flush()
    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="unit",
        trigger_source="upload",
        status="embedding",
        started_at=datetime.utcnow(),
        total_files=1,
    )
    db_session.add(batch)
    db_session.flush()
    job = IngestionJob(
        knowledge_base_id=sample_knowledge_base.id,
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
    new_chunk = _seed_active_chunk(db_session, sample_knowledge_base, document, new_version, content="new", snippet="new", chunk_version=2)
    old_chunk.state = "inactive"
    db_session.commit()

    request_batch_cancel(db_session, batch.id, sample_knowledge_base.id)
    stats = await compensate_cancelled_batch(db_session, batch.id)

    db_session.refresh(batch)
    db_session.refresh(old_version)
    db_session.refresh(new_version)
    db_session.refresh(old_chunk)
    db_session.refresh(new_chunk)
    assert batch.status == "cancelled"
    assert old_version.is_active is True
    assert old_chunk.state == "active"
    assert new_version.is_active is False
    assert new_chunk.state == "inactive"
    assert new_chunk.id in FakeVectorStore.deleted
    assert stats["restored_document_versions"] == 1
    db_session.refresh(sample_knowledge_base)
    assert sample_knowledge_base.current_chunk_version == 1


@pytest.mark.asyncio
@pytest.mark.skip(reason="Legacy run_graph_rebuild task was removed; evidence graph is built during ingestion.")
async def test_run_graph_rebuild_incremental_cancel_does_not_fallback_to_full(db_session, sample_knowledge_base, monkeypatch):
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
        knowledge_base_id=sample_knowledge_base.id,
        source_root="graph",
        trigger_source="rebuild_graph",
        status="completed",
        started_at=datetime.utcnow() - timedelta(hours=2),
        completed_at=datetime.utcnow() - timedelta(hours=1),
    )
    current_batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="graph",
        trigger_source="rebuild_graph",
        status="queued",
        total_files=0,
    )
    document = Document(
        knowledge_base_id=sample_knowledge_base.id,
        title="Changed",
        source_path="changed.md",
        source_type="markdown",
        checksum="changed",
        tags=[],
        is_active=True,
    )
    db_session.add_all([last_batch, current_batch, document])
    db_session.commit()

    async def fake_incremental_update(session, knowledge_base_id, changed_document_ids, batch_id, **_kwargs):
        assert changed_document_ids == [document.id]
        raise IngestionCancelled("unit cancel")

    async def forbidden_full_rebuild(*args, **kwargs):
        raise AssertionError("cancelled incremental rebuild must not fall back to full rebuild")

    monkeypatch.setattr(ingestion, "acquire_knowledge_base_graph_lock", lambda session, knowledge_base_id: True)
    monkeypatch.setattr(ingestion, "release_knowledge_base_graph_lock", lambda session, knowledge_base_id: None)
    monkeypatch.setattr(ingestion, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(ingestion, "rebuild_knowledge_base_graph", forbidden_full_rebuild)
    monkeypatch.setattr(concept_graph, "incremental_update_knowledge_base_graph", fake_incremental_update)
    monkeypatch.setattr(concept_graph, "_restore_knowledge_base_graph_from_backup", lambda session, knowledge_base_id: None)

    result = await run_graph_rebuild(current_batch.id, sample_knowledge_base.id, mode="incremental")

    db_session.expire_all()
    refreshed = db_session.get(IngestionBatch, current_batch.id)
    assert result["state"] == "cancelled"
    assert refreshed.status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Legacy run_graph_rebuild task was removed; evidence graph is built during ingestion.")
async def test_run_graph_rebuild_lock_failure_marks_batch_failed(db_session, sample_knowledge_base, monkeypatch):
    from app.models import IngestionBatch
    from app.services import ingestion
    from app.services.ingestion import run_graph_rebuild

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="graph",
        trigger_source="rebuild_graph",
        status="extracting_graph",
        total_files=0,
    )
    db_session.add(batch)
    db_session.commit()

    monkeypatch.setattr(ingestion, "acquire_knowledge_base_graph_lock", lambda session, knowledge_base_id: False)
    monkeypatch.setattr(ingestion, "GRAPH_LOCK_WAIT_SECONDS", 0)

    def forbidden_release(*args, **kwargs):
        raise AssertionError("must not release a graph lock this worker did not acquire")

    monkeypatch.setattr(ingestion, "release_knowledge_base_graph_lock", forbidden_release)

    with pytest.raises(RuntimeError, match="课程图谱正在重建中"):
        await run_graph_rebuild(batch.id, sample_knowledge_base.id, mode="full")

    db_session.expire_all()
    refreshed = db_session.get(IngestionBatch, batch.id)
    assert refreshed.status == "failed"
    assert refreshed.completed_at is not None
    assert refreshed.last_error == "课程图谱正在重建中，请等待当前任务完成"


@pytest.mark.asyncio
async def test_graph_phase_cancel_preserves_committed_parse_jobs(db_session, sample_knowledge_base, monkeypatch):
    from datetime import datetime

    from app.models import IngestionBatch, IngestionJob
    from app.services.ingestion import compensate_cancelled_batch, request_batch_cancel

    batch = IngestionBatch(
        knowledge_base_id=sample_knowledge_base.id,
        source_root="unit",
        trigger_source="upload",
        status="extracting_graph",
        started_at=datetime.utcnow(),
        total_files=1,
        processed_files=1,
        success_count=1,
        stats={"phase": "graph", "parse_committed": True},
    )
    db_session.add(batch)
    db_session.flush()
    job = IngestionJob(
        knowledge_base_id=sample_knowledge_base.id,
        batch_id=batch.id,
        source_path="parsed.md",
        trigger_source="upload",
        status="completed",
        stats={"chunks": 2},
    )
    db_session.add(job)
    db_session.commit()

    request_batch_cancel(db_session, batch.id, sample_knowledge_base.id)
    stats = await compensate_cancelled_batch(db_session, batch.id)

    db_session.refresh(batch)
    db_session.refresh(job)
    assert batch.status == "cancelled"
    assert job.status == "completed"
    assert stats["parse_rollback_skipped"] is True


@pytest.mark.asyncio
async def test_uploaded_parse_records_evidence_graph_runtime(db_session, sample_knowledge_base, monkeypatch):
    from app.services import ingestion
    from app.services.ingestion import create_uploaded_files_batch, run_uploaded_files_ingestion

    from app.core.config import get_settings

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"]
    source_path = storage_root / "note.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Note\n\nFresh content", encoding="utf-8")

    batch = create_uploaded_files_batch(db_session, sample_knowledge_base.id, [source_path], force=False)

    async def fake_ingest_file(*args, **kwargs):
        return {
            "status": "completed",
            "source_type": "markdown",
            "stats": {"evidence_atoms": 2, "evidence_edges": 1, "active_chunks": 1},
        }

    monkeypatch.setattr(ingestion, "ingest_file", fake_ingest_file)

    result = await run_uploaded_files_ingestion(batch.id, [str(source_path)], force=False)

    assert result["state"] == "completed"
    assert result["graph_stats"]["graph_runtime"] == "evidence_graph"
    assert result["graph_stats"]["graph_rebuilt"] is True


@pytest.mark.asyncio
async def test_uploaded_parse_marks_parse_failure_terminal(db_session, sample_knowledge_base, monkeypatch):
    from app.core.config import get_settings
    from app.models import IngestionBatch
    from app.services import ingestion
    from app.services.ingestion import create_uploaded_files_batch, run_uploaded_files_ingestion

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"]
    source_path = storage_root / "graph-failure.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Note\n\nFresh content", encoding="utf-8")

    batch = create_uploaded_files_batch(db_session, sample_knowledge_base.id, [source_path], force=False)

    async def fake_ingest_file(*args, **kwargs):
        raise RuntimeError("parse unavailable")

    monkeypatch.setattr(ingestion, "ingest_file", fake_ingest_file)

    result = await run_uploaded_files_ingestion(batch.id, [str(source_path)], force=False)
    db_session.expire_all()
    refreshed = db_session.get(IngestionBatch, batch.id)

    assert result["state"] == "failed"
    assert refreshed.status == "failed"
    assert refreshed.completed_at is not None
    assert "parse unavailable" in refreshed.last_error


@pytest.mark.asyncio
async def test_ingest_short_markdown_keeps_parent_and_child_chunks(db_session, sample_knowledge_base, monkeypatch):
    from app.core.config import get_settings
    from app.models import ActiveChunk
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

        def health_check(self, knowledge_base_id, active_chunk_ids):
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

    storage_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["storage_root"]
    source_path = storage_root / "short-centrality.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "# Centrality smoke test\n\n"
        "Degree centrality counts incident edges in a graph. "
        "It is a local network-analysis measure used to compare node prominence.",
        encoding="utf-8",
    )

    result = await ingest_file(db_session, source_path, knowledge_base_id=sample_knowledge_base.id, rebuild_graph=False, force=True)

    chunks = [
        chunk
        for chunk in db_session.query(ActiveChunk).filter(ActiveChunk.knowledge_base_id == sample_knowledge_base.id, ActiveChunk.state == "active").all()
        if (chunk.metadata_json or {}).get("document_id") == result["document_id"]
    ]
    assert any((chunk.metadata_json or {}).get("is_parent") for chunk in chunks)
    assert any(not (chunk.metadata_json or {}).get("is_parent") for chunk in chunks)
    assert result["stats"]["active_chunks"] >= 2

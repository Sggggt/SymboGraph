from __future__ import annotations

import pytest


def add_active_document_graph(db_session, knowledge_base_id: str, *, active: bool, title: str = "Doc") -> dict:
    from app.models import (
        ActiveChunk,
        ChunkCandidate,
        ChunkDecision,
        Document,
        DocumentVersion,
        EvidenceAtom,
        EvidenceGraphState,
        PolicyState,
        QualityDecision,
    )

    slug = title.lower().replace(" ", "-")
    document = Document(
        knowledge_base_id=knowledge_base_id,
        title=title,
        source_path=f"{slug}.md",
        source_type="markdown",
        checksum=f"{slug}-checksum",
        is_active=active,
    )
    db_session.add(document)
    db_session.flush()
    version = DocumentVersion(
        document_id=document.id,
        version=1,
        checksum=document.checksum,
        storage_path=document.source_path,
        is_active=active,
    )
    db_session.add(version)
    db_session.flush()
    policy = PolicyState(
        knowledge_base_id=knowledge_base_id,
        profile_objective_hash=f"{slug}-objective",
        posterior_json={},
        constraints_json={},
        exploration_json={},
        reward_summary_json={},
        state_hash=f"{slug}-policy",
    )
    db_session.add(policy)
    db_session.flush()
    text = f"{title} content about network centrality and graph evidence."
    atom = EvidenceAtom(
        knowledge_base_id=knowledge_base_id,
        document_id=document.id,
        document_version_id=version.id,
        atom_index=0,
        atom_type="paragraph",
        text=text,
        text_hash=f"{slug}-atom-hash",
        source_span_json={"spans": [{"start": 0, "end": len(text), "section": "Section"}]},
        metadata_json={"section": "Section"},
        state="active" if active else "inactive",
    )
    db_session.add(atom)
    db_session.flush()
    graph_state = EvidenceGraphState(
        knowledge_base_id=knowledge_base_id,
        scope_type="document",
        state_hash=f"{slug}-graph",
        atom_scope_hash=f"{slug}-atom-scope",
        active_document_version_ids=[version.id],
        active_atom_ids=[atom.id],
        policy_state_id=policy.id,
        state="active" if active else "inactive",
    )
    db_session.add(graph_state)
    db_session.flush()
    candidate = ChunkCandidate(
        graph_state_id=graph_state.id,
        generator_name="maintenance-test",
        generator_version="v1",
        atom_ids_json=[atom.id],
        source_span_union_json={"spans": atom.source_span_json["spans"]},
        token_count=12,
        graph_features_json={"fixture": True},
    )
    db_session.add(candidate)
    db_session.flush()
    quality = QualityDecision(
        candidate_id=candidate.id,
        policy_state_id=policy.id,
        decision_action="answer_candidate",
        gate_passed=True,
    )
    db_session.add(quality)
    db_session.flush()
    decision = ChunkDecision(
        knowledge_base_id=knowledge_base_id,
        graph_state_id=graph_state.id,
        candidate_id=candidate.id,
        quality_decision_id=quality.id,
        policy_state_id=policy.id,
        action="activate",
    )
    db_session.add(decision)
    db_session.flush()
    active_chunk = ActiveChunk(
        knowledge_base_id=knowledge_base_id,
        chunk_decision_id=decision.id,
        document_version_scope_hash=f"{slug}-version-scope",
        graph_state_hash=graph_state.state_hash,
        atom_ids_json=[atom.id],
        text=text,
        source_span_union_json={"spans": atom.source_span_json["spans"]},
        boundary_policy_version="maintenance-test-v1",
        quality_decision_id=quality.id,
        policy_state_id=policy.id,
        metadata_json={
            "document_id": document.id,
            "document_version_id": version.id,
            "chunk_version": 1,
            "partition": "Unit",
            "section": "Section",
            "source_type": "markdown",
            "content_kind": "markdown",
            "snippet": f"{title} content",
        },
        state="active" if active else "inactive",
    )
    db_session.add(active_chunk)
    db_session.flush()
    return {
        "document": document,
        "version": version,
        "policy": policy,
        "atom": atom,
        "graph_state": graph_state,
        "candidate": candidate,
        "quality": quality,
        "decision": decision,
        "chunk": active_chunk,
    }


def test_cleanup_stale_data_deletes_only_inactive_rows_and_stale_vectors(db_session, sample_knowledge_base, monkeypatch):
    from app.models import (
        ActiveChunk,
        ChunkCandidate,
        ChunkDecision,
        CommunityMembership,
        CommunityState,
        CommunitySummary,
        Document,
        DocumentVersion,
        EvidenceAtom,
        EvidenceEdge,
        EvidenceGraphState,
        IngestionJob,
        QualityDecision,
        VectorRecord,
    )
    from app.services import maintenance
    from app.services.maintenance import cleanup_stale_data

    active = add_active_document_graph(db_session, sample_knowledge_base.id, active=True, title="Active")
    inactive = add_active_document_graph(db_session, sample_knowledge_base.id, active=False, title="Inactive")
    stale_edge = EvidenceEdge(
        graph_state_id=inactive["graph_state"].id,
        source_atom_id=inactive["atom"].id,
        target_atom_id=inactive["atom"].id,
        edge_type="adjacent",
    )
    stale_community_state = CommunityState(
        knowledge_base_id=sample_knowledge_base.id,
        graph_state_id=inactive["graph_state"].id,
        state_hash="stale-community",
    )
    job = IngestionJob(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=inactive["document"].id,
        source_path=inactive["document"].source_path,
        trigger_source="remove",
        status="skipped",
    )
    db_session.add_all([stale_edge, stale_community_state, job])
    db_session.flush()
    stale_membership = CommunityMembership(
        community_state_id=stale_community_state.id,
        community_id="community-1",
        atom_id=inactive["atom"].id,
    )
    stale_summary = CommunitySummary(
        community_state_id=stale_community_state.id,
        community_id="community-1",
        summary="stale summary",
        evidence_atom_ids_json=[inactive["atom"].id],
    )
    stale_record = VectorRecord(
        knowledge_base_id=sample_knowledge_base.id,
        active_chunk_id=inactive["chunk"].id,
        qdrant_point_id=inactive["chunk"].id,
        embedding_model="test-embedding",
        embedding_text_version="metadata_enriched_v1",
        payload_hash="stale-payload",
    )
    db_session.add_all([stale_membership, stale_summary, stale_record])
    db_session.commit()

    active_document_id = active["document"].id
    active_version_id = active["version"].id
    active_chunk_id = active["chunk"].id
    active_atom_id = active["atom"].id
    active_graph_state_id = active["graph_state"].id
    inactive_document_id = inactive["document"].id
    inactive_version_id = inactive["version"].id
    inactive_chunk_id = inactive["chunk"].id
    inactive_atom_id = inactive["atom"].id
    stale_graph_state_id = inactive["graph_state"].id
    stale_record_id = stale_record.id
    job_id = job.id

    deleted_vectors: list[str] = []

    class TrackingVectorStore:
        def __init__(self, knowledge_base_name=None):
            self.knowledge_base_name = knowledge_base_name

        def list_ids(self, knowledge_base_id=None):
            return [active_chunk_id, inactive_chunk_id, "qdrant-only-stale"]

        def delete(self, ids):
            deleted_vectors.extend(ids)

    monkeypatch.setattr(maintenance, "VectorStore", TrackingVectorStore)

    stats = cleanup_stale_data(db_session, sample_knowledge_base.id, sample_knowledge_base.name)

    assert stats["deleted_vectors"] == 2
    assert set(deleted_vectors) == {inactive_chunk_id, "qdrant-only-stale"}
    assert stats["deleted_chunks"] == 1
    assert stats["deleted_document_versions"] == 1
    assert stats["deleted_documents"] == 1
    assert stats["removed_vector_records"] == 1
    assert stats["removed_evidence_atoms"] == 1
    assert stats["removed_evidence_edges"] == 1
    assert stats["removed_evidence_graph_states"] == 1
    assert stats["removed_active_chunks"] == 1
    assert stats["removed_chunk_candidates"] == 1
    assert stats["removed_chunk_decisions"] == 1
    assert stats["removed_quality_decisions"] == 1
    assert stats["removed_community_states"] == 1
    assert stats["removed_community_memberships"] == 1
    assert stats["removed_community_summaries"] == 1
    assert db_session.get(Document, active_document_id) is not None
    assert db_session.get(DocumentVersion, active_version_id) is not None
    assert db_session.get(ActiveChunk, active_chunk_id) is not None
    assert db_session.get(EvidenceAtom, active_atom_id) is not None
    assert db_session.get(EvidenceGraphState, active_graph_state_id) is not None
    assert db_session.get(Document, inactive_document_id) is None
    assert db_session.get(DocumentVersion, inactive_version_id) is None
    assert db_session.get(ActiveChunk, inactive_chunk_id) is None
    assert db_session.get(EvidenceAtom, inactive_atom_id) is None
    assert db_session.get(EvidenceGraphState, stale_graph_state_id) is None
    assert db_session.get(VectorRecord, stale_record_id) is None
    assert db_session.query(ChunkDecision).filter_by(id=inactive["decision"].id).count() == 0
    assert db_session.query(ChunkCandidate).filter_by(id=inactive["candidate"].id).count() == 0
    assert db_session.query(QualityDecision).filter_by(id=inactive["quality"].id).count() == 0
    refreshed_job = db_session.get(IngestionJob, job_id)
    assert refreshed_job is not None
    assert refreshed_job.document_id is None


def test_cleanup_stale_graph_is_evidence_only_noop(db_session, sample_knowledge_base):
    from app.services.maintenance import cleanup_stale_graph

    add_active_document_graph(db_session, sample_knowledge_base.id, active=True, title="GraphActive")
    db_session.commit()

    stats = cleanup_stale_graph(db_session, sample_knowledge_base.id)

    assert stats == {
        "removed_evidence_edges": 0,
        "removed_evidence_atoms": 0,
        "removed_signal_nodes": 0,
        "removed_signal_edges": 0,
    }


def test_reconcile_vector_store_orphans_deletes_qdrant_only_points(db_session, sample_knowledge_base, monkeypatch):
    from app.models import ActiveChunk, VectorRecord
    from app.services import maintenance
    from app.services.maintenance import reconcile_vector_store_orphans

    active = add_active_document_graph(db_session, sample_knowledge_base.id, active=True, title="VectorActive")
    active_chunk_id = active["chunk"].id
    orphan_id = "qdrant-only-orphan"
    orphan_record = VectorRecord(
        knowledge_base_id=sample_knowledge_base.id,
        active_chunk_id=active_chunk_id,
        qdrant_point_id=orphan_id,
        embedding_model="unit-test",
        embedding_text_version="unit-v1",
        payload_hash="orphan",
        vector_status="ready",
    )
    db_session.add(orphan_record)
    db_session.commit()

    deleted_vectors: list[str] = []

    class TrackingVectorStore:
        def __init__(self, knowledge_base_name=None):
            self.knowledge_base_name = knowledge_base_name

        def list_ids(self, knowledge_base_id=None):
            return [active_chunk_id, orphan_id]

        def delete(self, ids):
            deleted_vectors.extend(ids)

    monkeypatch.setattr(maintenance, "VectorStore", TrackingVectorStore)

    stats = reconcile_vector_store_orphans(db_session, sample_knowledge_base.id)

    assert stats["deleted_orphan_points"] == 1
    assert stats["removed_orphan_vector_records"] == 1
    assert stats["missing_points"] == 0
    assert deleted_vectors == [orphan_id]
    assert db_session.get(ActiveChunk, active_chunk_id) is not None
    assert db_session.query(VectorRecord).filter(VectorRecord.id == orphan_record.id).count() == 0


def test_delete_document_graph_incremental_removes_document_evidence_scope(db_session, sample_knowledge_base):
    from app.models import ActiveChunk, Document, EvidenceAtom, EvidenceGraphState
    from app.services.maintenance import delete_document_graph_incremental

    keep = add_active_document_graph(db_session, sample_knowledge_base.id, active=True, title="KeepDoc")
    delete = add_active_document_graph(db_session, sample_knowledge_base.id, active=True, title="DeleteDoc")
    db_session.commit()

    keep_chunk_id = keep["chunk"].id
    keep_atom_id = keep["atom"].id
    delete_document_id = delete["document"].id
    delete_chunk_id = delete["chunk"].id
    delete_atom_id = delete["atom"].id
    delete_graph_state_id = delete["graph_state"].id

    stats = delete_document_graph_incremental(db_session, sample_knowledge_base.id, delete_document_id)
    db_session.commit()

    assert stats["removed_active_chunks"] == 1
    assert stats["removed_evidence_atoms"] == 1
    assert stats["removed_evidence_graph_states"] == 1
    assert db_session.get(ActiveChunk, keep_chunk_id) is not None
    assert db_session.get(EvidenceAtom, keep_atom_id) is not None
    assert db_session.get(ActiveChunk, delete_chunk_id) is None
    assert db_session.get(EvidenceAtom, delete_atom_id) is None
    assert db_session.get(EvidenceGraphState, delete_graph_state_id) is None
    assert db_session.get(Document, delete_document_id) is not None


def test_cleanup_stale_data_rejects_active_batch(db_session, sample_knowledge_base):
    from app.models import IngestionBatch
    from app.services.maintenance import MaintenanceConflict, cleanup_stale_data

    db_session.add(IngestionBatch(knowledge_base_id=sample_knowledge_base.id, source_root="unit", trigger_source="upload", status="queued"))
    db_session.commit()

    with pytest.raises(MaintenanceConflict):
        cleanup_stale_data(db_session, sample_knowledge_base.id, sample_knowledge_base.name)


@pytest.mark.asyncio
async def test_rebuild_graph_endpoint_reports_evidence_status(db_session, sample_knowledge_base, monkeypatch):
    from app.api import rebuild_graph_endpoint
    from app.core.config import get_settings
    from app.schemas import RebuildGraphRequest

    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    monkeypatch.setenv("INGESTION_EXECUTION_MODE", "inline")
    get_settings.cache_clear()

    response = await rebuild_graph_endpoint(
        request=RebuildGraphRequest(mode="evidence"),
        knowledge_base_id=sample_knowledge_base.id,
        db=db_session,
    )

    assert response["mode"] == "evidence"
    assert response["batch_id"] is None
    assert response["state"] == "evidence_graph_active"
    assert response["affected_documents"] >= 0


@pytest.mark.asyncio
async def test_rebuild_graph_endpoint_dry_run_has_no_batch(db_session, sample_knowledge_base, monkeypatch):
    from app.api import rebuild_graph_endpoint
    from app.core.config import get_settings
    from app.schemas import RebuildGraphRequest

    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    get_settings.cache_clear()

    response = await rebuild_graph_endpoint(
        request=RebuildGraphRequest(mode="evidence", dry_run=True),
        knowledge_base_id=sample_knowledge_base.id,
        db=db_session,
    )

    assert response["batch_id"] is None
    assert response["state"] == "evidence_graph_active"
    assert response["mode"] == "evidence"
    assert response["dry_run"] is True
    assert response["affected_documents"] >= 0


def test_delete_knowledge_base_data_removes_database_vectors_and_directory(db_session, sample_knowledge_base, monkeypatch):
    from app.core.config import get_settings
    from app.models import (
        AgentRun,
        AgentTraceEvent,
        AnswerSession,
        CitationVerification,
        IngestionBatch,
        IngestionCompensationLog,
        IngestionJob,
        IngestionLog,
        KnowledgeBase,
        ParseJob,
        PolicyObservation,
        QASession,
        QualityObservation,
        RetrievalTrace,
        RewardEvent,
        SourceFile,
        VectorRecord,
    )
    from app.services import maintenance
    from app.services.maintenance import delete_knowledge_base_data

    bundle = add_active_document_graph(db_session, sample_knowledge_base.id, active=True, title="DeleteMe")
    batch = IngestionBatch(knowledge_base_id=sample_knowledge_base.id, source_root="unit", trigger_source="upload", status="completed")
    db_session.add(batch)
    db_session.flush()
    job = IngestionJob(
        knowledge_base_id=sample_knowledge_base.id,
        batch_id=batch.id,
        document_id=bundle["document"].id,
        source_path=bundle["document"].source_path,
        status="completed",
    )
    qa_session = QASession(knowledge_base_id=sample_knowledge_base.id, title="chat", transcript=[])
    run = AgentRun(knowledge_base_id=sample_knowledge_base.id, question="q", status="completed")
    retrieval_trace = RetrievalTrace(
        knowledge_base_id=sample_knowledge_base.id,
        query="q",
        result_active_chunk_ids_json=[bundle["chunk"].id],
    )
    db_session.add_all([job, qa_session, run, retrieval_trace])
    db_session.flush()
    answer_session = AnswerSession(
        knowledge_base_id=sample_knowledge_base.id,
        retrieval_trace_id=retrieval_trace.id,
        qa_session_id=qa_session.id,
        question="q",
        answer="a",
        active_chunk_ids_json=[bundle["chunk"].id],
    )
    db_session.add(answer_session)
    db_session.flush()
    source_file = SourceFile(
        knowledge_base_id=sample_knowledge_base.id,
        document_id=bundle["document"].id,
        source_path=bundle["document"].source_path,
        checksum=bundle["document"].checksum,
        source_type="markdown",
    )
    db_session.add(source_file)
    db_session.flush()
    db_session.add_all(
        [
            IngestionLog(batch_id=batch.id, event="done", message="done"),
            IngestionCompensationLog(
                knowledge_base_id=sample_knowledge_base.id,
                job_id=job.id,
                operation="delete",
                vector_ids=[bundle["chunk"].id],
                status="completed",
            ),
            ParseJob(
                knowledge_base_id=sample_knowledge_base.id,
                document_id=bundle["document"].id,
                document_version_id=bundle["version"].id,
                ingestion_job_id=job.id,
                source_file_id=source_file.id,
            ),
            AgentTraceEvent(run_id=run.id, node="n", status="completed", document_ids=[], scores={}),
            CitationVerification(
                knowledge_base_id=sample_knowledge_base.id,
                answer_session_id=answer_session.id,
                retrieval_trace_id=retrieval_trace.id,
                active_chunk_id=bundle["chunk"].id,
                claim_text="claim",
            ),
            QualityObservation(
                knowledge_base_id=sample_knowledge_base.id,
                quality_decision_id=bundle["quality"].id,
                observation_type="gate",
                observation_json={"ok": True},
            ),
            PolicyObservation(
                knowledge_base_id=sample_knowledge_base.id,
                policy_state_id=bundle["policy"].id,
                context_json={},
                action_json={},
                reward_json={"score": 1.0},
            ),
            RewardEvent(
                knowledge_base_id=sample_knowledge_base.id,
                policy_state_id=bundle["policy"].id,
                retrieval_trace_id=retrieval_trace.id,
                answer_session_id=answer_session.id,
                active_chunk_ids_json=[bundle["chunk"].id],
                reward_json={"score": 1.0},
            ),
            VectorRecord(
                knowledge_base_id=sample_knowledge_base.id,
                active_chunk_id=bundle["chunk"].id,
                qdrant_point_id=bundle["chunk"].id,
                embedding_model="test-embedding",
                embedding_text_version="metadata_enriched_v1",
                payload_hash="payload",
            ),
        ]
    )
    db_session.commit()

    knowledge_base_root = get_settings().knowledge_base_paths_for_name(sample_knowledge_base.name)["knowledge_base_root"]
    (knowledge_base_root / "storage").mkdir(parents=True, exist_ok=True)
    (knowledge_base_root / "storage" / "note.md").write_text("delete me", encoding="utf-8")
    deleted_vectors: list[str] = []
    operations: list[str] = []
    chunk_id = bundle["chunk"].id

    class TrackingVectorStore:
        def __init__(self, knowledge_base_name=None):
            self.knowledge_base_name = knowledge_base_name

        def list_ids(self, knowledge_base_id=None):
            return [chunk_id, "orphan-vector"]

        def delete(self, ids):
            operations.append("qdrant_delete")
            deleted_vectors.extend(ids)

    monkeypatch.setattr(maintenance, "VectorStore", TrackingVectorStore)
    original_commit = db_session.commit

    def tracking_commit():
        operations.append("db_commit")
        original_commit()

    monkeypatch.setattr(db_session, "commit", tracking_commit)
    knowledge_base_id = sample_knowledge_base.id

    stats = delete_knowledge_base_data(db_session, sample_knowledge_base)

    assert stats["deleted_vectors"] == 2
    assert set(deleted_vectors) == {chunk_id, "orphan-vector"}
    assert stats["deleted_vector_records"] == 1
    assert stats["deleted_active_chunks"] == 1
    assert stats["deleted_chunk_decisions"] == 1
    assert stats["deleted_quality_decisions"] == 1
    assert stats["deleted_chunk_candidates"] == 1
    assert stats["deleted_evidence_atoms"] == 1
    assert stats["deleted_evidence_graph_states"] == 1
    assert stats["deleted_policy_states"] == 1
    assert stats["deleted_retrieval_traces"] == 1
    assert stats["deleted_answer_sessions"] == 1
    assert stats["deleted_citation_verifications"] == 1
    assert stats["deleted_reward_events"] == 1
    assert stats["deleted_quality_observations"] == 1
    assert stats["deleted_policy_observations"] == 1
    assert stats["deleted_parse_jobs"] == 1
    assert stats["deleted_source_files"] == 1
    assert stats["deleted_ingestion_logs"] == 1
    assert stats["deleted_compensations"] == 1
    assert stats["deleted_jobs"] == 1
    assert stats["deleted_batches"] == 1
    assert stats["deleted_knowledge_bases"] == 1
    assert stats["deleted_directory"] == 1
    assert operations.index("db_commit") < operations.index("qdrant_delete")
    assert not knowledge_base_root.exists()
    assert db_session.get(KnowledgeBase, knowledge_base_id) is None


def test_cleanup_stale_data_commits_db_before_qdrant_delete(db_session, sample_knowledge_base, monkeypatch):
    from app.services import maintenance
    from app.services.maintenance import cleanup_stale_data

    operations: list[str] = []

    class TrackingVectorStore:
        def __init__(self, knowledge_base_name=None):
            self.knowledge_base_name = knowledge_base_name

        def list_ids(self, knowledge_base_id=None):
            return ["stale-vector-1"]

        def delete(self, ids):
            operations.append("qdrant_delete")

    original_commit = db_session.commit

    def tracking_commit():
        operations.append("db_commit")
        original_commit()

    monkeypatch.setattr(maintenance, "VectorStore", TrackingVectorStore)
    monkeypatch.setattr(db_session, "commit", tracking_commit)

    cleanup_stale_data(db_session, sample_knowledge_base.id, sample_knowledge_base.name)

    db_commit_indices = [index for index, op in enumerate(operations) if op == "db_commit"]
    qdrant_index = operations.index("qdrant_delete")
    assert any(index < qdrant_index for index in db_commit_indices), (
        f"DB commit must occur before Qdrant delete, but operations were: {operations}"
    )

from __future__ import annotations

import pytest


def add_document_graph(db_session, course_id: str, *, active: bool, title: str = "Doc"):
    from app.models import Chunk, Document, DocumentVersion

    document = Document(
        course_id=course_id,
        title=title,
        source_path=f"{title.lower()}.md",
        source_type="markdown",
        checksum=f"{title}-checksum",
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
    chunk = Chunk(
        course_id=course_id,
        document_id=document.id,
        document_version_id=version.id,
        content=f"{title} content about network centrality and graph evidence.",
        snippet=f"{title} content",
        chapter="Unit",
        section="Section",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=active,
    )
    db_session.add(chunk)
    db_session.flush()
    return document, version, chunk


def add_concept(db_session, course_id: str, name: str):
    from app.models import Concept, ConceptAlias

    concept = Concept(
        course_id=course_id,
        canonical_name=name,
        normalized_name=name.lower(),
        summary=name,
        importance_score=0.5,
    )
    db_session.add(concept)
    db_session.flush()
    db_session.add(ConceptAlias(concept_id=concept.id, alias=name, normalized_alias=name.lower()))
    db_session.flush()
    return concept


def test_cleanup_stale_data_deletes_only_inactive_rows_and_stale_vectors(db_session, sample_course, monkeypatch):
    from app.models import Chunk, Concept, ConceptAlias, ConceptRelation, Document, DocumentVersion, IngestionJob
    from app.services import maintenance
    from app.services.maintenance import cleanup_stale_data

    active_document, active_version, active_chunk = add_document_graph(db_session, sample_course.id, active=True, title="Active")
    inactive_document, inactive_version, inactive_chunk = add_document_graph(db_session, sample_course.id, active=False, title="Inactive")
    stale_source = add_concept(db_session, sample_course.id, "Stale Source")
    stale_target = add_concept(db_session, sample_course.id, "Stale Target")
    active_source = add_concept(db_session, sample_course.id, "Active Source")
    active_target = add_concept(db_session, sample_course.id, "Active Target")
    stale_relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=stale_source.id,
        target_concept_id=stale_target.id,
        target_name=stale_target.canonical_name,
        relation_type="related_to",
        evidence_chunk_id=inactive_chunk.id,
    )
    active_relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=active_source.id,
        target_concept_id=active_target.id,
        target_name=active_target.canonical_name,
        relation_type="related_to",
        evidence_chunk_id=active_chunk.id,
    )
    job = IngestionJob(
        course_id=sample_course.id,
        document_id=inactive_document.id,
        source_path=inactive_document.source_path,
        trigger_source="remove",
        status="skipped",
    )
    db_session.add_all([stale_relation, active_relation, job])
    db_session.commit()
    inactive_document_id = inactive_document.id
    inactive_version_id = inactive_version.id
    inactive_chunk_id = inactive_chunk.id
    stale_relation_id = stale_relation.id
    active_relation_id = active_relation.id
    stale_source_id = stale_source.id
    stale_source_alias_id = stale_source.aliases[0].id

    deleted_vectors: list[str] = []

    class TrackingVectorStore:
        def __init__(self, course_name=None):
            self.course_name = course_name

        def list_ids(self, course_id=None):
            return [active_chunk.id, inactive_chunk_id, "qdrant-only-stale"]

        def delete(self, ids):
            deleted_vectors.extend(ids)

    monkeypatch.setattr(maintenance, "VectorStore", TrackingVectorStore)

    stats = cleanup_stale_data(db_session, sample_course.id, sample_course.name)

    assert stats["deleted_vectors"] == 2
    assert set(deleted_vectors) == {inactive_chunk_id, "qdrant-only-stale"}
    assert stats["deleted_chunks"] == 1
    assert stats["deleted_document_versions"] == 1
    assert stats["deleted_documents"] == 1
    assert stats["removed_graph_relations"] == 1
    assert stats["removed_graph_concepts"] == 2
    assert db_session.get(Document, active_document.id) is not None
    assert db_session.get(DocumentVersion, active_version.id) is not None
    assert db_session.get(Chunk, active_chunk.id) is not None
    assert db_session.get(Document, inactive_document_id) is None
    assert db_session.get(DocumentVersion, inactive_version_id) is None
    assert db_session.get(Chunk, inactive_chunk_id) is None
    assert db_session.get(ConceptRelation, active_relation_id) is not None
    assert db_session.get(ConceptRelation, stale_relation_id) is None
    assert db_session.get(Concept, stale_source_id) is None
    assert db_session.get(ConceptAlias, stale_source_alias_id) is None
    refreshed_job = db_session.get(IngestionJob, job.id)
    assert refreshed_job is not None
    assert refreshed_job.document_id is None


def test_cleanup_stale_graph_removes_invalid_relations_and_orphans(db_session, sample_course):
    from app.models import Concept, ConceptRelation
    from app.services.maintenance import cleanup_stale_graph

    active_document, _, active_chunk = add_document_graph(db_session, sample_course.id, active=True, title="GraphActive")
    inactive_document, _, inactive_chunk = add_document_graph(db_session, sample_course.id, active=False, title="GraphInactive")
    keep_source = add_concept(db_session, sample_course.id, "Keep Source")
    keep_target = add_concept(db_session, sample_course.id, "Keep Target")
    stale_source = add_concept(db_session, sample_course.id, "Dead Source")
    stale_target = add_concept(db_session, sample_course.id, "Dead Target")
    keep_relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=keep_source.id,
        target_concept_id=keep_target.id,
        target_name=keep_target.canonical_name,
        relation_type="defines",
        evidence_chunk_id=active_chunk.id,
    )
    stale_relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=stale_source.id,
        target_concept_id=stale_target.id,
        target_name=stale_target.canonical_name,
        relation_type="supports",
        evidence_chunk_id=inactive_chunk.id,
    )
    db_session.add_all([keep_relation, stale_relation])
    db_session.commit()
    keep_relation_id = keep_relation.id
    stale_relation_id = stale_relation.id
    keep_source_id = keep_source.id
    keep_target_id = keep_target.id
    stale_source_id = stale_source.id
    stale_target_id = stale_target.id

    stats = cleanup_stale_graph(db_session, sample_course.id)

    assert stats == {"removed_relations": 1, "removed_aliases": 2, "removed_concepts": 2, "migrated_relations": 1}
    assert db_session.get(ConceptRelation, keep_relation_id) is not None
    assert db_session.get(ConceptRelation, keep_relation_id).relation_type == "defined_by"
    assert db_session.get(ConceptRelation, stale_relation_id) is None
    assert db_session.get(Concept, keep_source_id) is not None
    assert db_session.get(Concept, keep_target_id) is not None
    assert db_session.get(Concept, stale_source_id) is None
    assert db_session.get(Concept, stale_target_id) is None
    assert active_document.is_active is True
    assert inactive_document.is_active is False


def test_delete_document_graph_incremental_gc_removes_only_orphan_facts(db_session, sample_course):
    from app.models import Concept, ConceptRelation, EntityMention
    from app.services.maintenance import delete_document_graph_incremental

    document_a, _, chunk_a = add_document_graph(db_session, sample_course.id, active=True, title="DocA")
    document_b, _, chunk_b = add_document_graph(db_session, sample_course.id, active=True, title="DocB")
    unique = add_concept(db_session, sample_course.id, "Unique Concept")
    shared = add_concept(db_session, sample_course.id, "Shared Concept")
    anchor = add_concept(db_session, sample_course.id, "Anchor Concept")
    orphan = add_concept(db_session, sample_course.id, "Empty Source Orphan")
    unique.source_document_ids = [document_a.id]
    unique.evidence_count = 1
    shared.source_document_ids = [document_a.id, document_b.id]
    shared.evidence_count = 2
    anchor.source_document_ids = [document_b.id]
    anchor.evidence_count = 1
    orphan.source_document_ids = []
    orphan.evidence_count = 0
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=unique.id,
            target_concept_id=shared.id,
            target_name=shared.canonical_name,
            relation_type="related_to",
            evidence_chunk_id=chunk_a.id,
            source_document_ids=[document_a.id],
        )
    )
    db_session.add(
        ConceptRelation(
            course_id=sample_course.id,
            source_concept_id=shared.id,
            target_concept_id=anchor.id,
            target_name=anchor.canonical_name,
            relation_type="related_to",
            source_document_ids=[document_a.id, document_b.id],
        )
    )
    db_session.add(
        EntityMention(
            course_id=sample_course.id,
            chunk_id=chunk_a.id,
            document_id=document_a.id,
            concept_id=unique.id,
            surface="Unique Concept",
            canonical_name="Unique Concept",
            normalized_key="unique concept::concept",
            entity_type="concept",
        )
    )
    for surface in ["Shared Concept", "shared concept"]:
        db_session.add(
            EntityMention(
                course_id=sample_course.id,
                chunk_id=chunk_b.id,
                document_id=document_b.id,
                concept_id=shared.id,
                surface=surface,
                canonical_name="Shared Concept",
                normalized_key="shared concept::concept",
                entity_type="concept",
            )
        )
    db_session.commit()
    unique_id = unique.id
    shared_id = shared.id
    orphan_id = orphan.id

    stats = delete_document_graph_incremental(db_session, sample_course.id, document_a.id)
    db_session.commit()
    db_session.expire_all()

    assert stats["graph_removed_concepts"] == 2
    assert db_session.get(Concept, unique_id) is None
    assert db_session.get(Concept, orphan_id) is None
    refreshed_shared = db_session.get(Concept, shared_id)
    assert refreshed_shared is not None
    assert refreshed_shared.source_document_ids == [document_b.id]
    assert refreshed_shared.evidence_count == 2
    remaining_relations = db_session.query(ConceptRelation).filter_by(course_id=sample_course.id).all()
    assert len(remaining_relations) == 1
    assert remaining_relations[0].source_document_ids == [document_b.id]
    assert db_session.query(EntityMention).filter_by(document_id=document_a.id).count() == 0


def test_delete_document_graph_incremental_recalculates_edge_support_from_remaining_mentions(db_session, sample_course):
    from app.models import Chunk, ConceptRelation, EntityMention, GraphRelationCandidate
    from app.services.maintenance import delete_document_graph_incremental

    document_a, _, chunk_a = add_document_graph(db_session, sample_course.id, active=True, title="SupportA")
    document_b, version_b, chunk_b1 = add_document_graph(db_session, sample_course.id, active=True, title="SupportB")
    chunk_b2 = Chunk(
        course_id=sample_course.id,
        document_id=document_b.id,
        document_version_id=version_b.id,
        content="SupportB second chunk with shared relation evidence.",
        snippet="SupportB second chunk",
        chapter="Unit",
        section="Section 2",
        source_type="markdown",
        metadata_json={"content_kind": "markdown"},
        embedding_status="ready",
        is_active=True,
    )
    db_session.add(chunk_b2)
    db_session.flush()
    source = add_concept(db_session, sample_course.id, "Shared Source")
    target = add_concept(db_session, sample_course.id, "Shared Target")
    relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=source.id,
        target_concept_id=target.id,
        target_name=target.canonical_name,
        relation_type="related_to",
        evidence_chunk_id=chunk_a.id,
        source_document_ids=[document_a.id, document_b.id],
        support_count=9,
    )
    candidate = GraphRelationCandidate(
        course_id=sample_course.id,
        source_concept_id=source.id,
        target_concept_id=target.id,
        target_name=target.canonical_name,
        relation_type="related_to",
        evidence_chunk_id=chunk_a.id,
        source_document_ids=[document_a.id, document_b.id],
        support_count=9,
    )
    db_session.add_all([relation, candidate])
    for chunk in (chunk_b1, chunk_b2):
        for concept, name in ((source, "Shared Source"), (target, "Shared Target")):
            db_session.add(
                EntityMention(
                    course_id=sample_course.id,
                    chunk_id=chunk.id,
                    document_id=document_b.id,
                    concept_id=concept.id,
                    surface=name,
                    canonical_name=name,
                    normalized_key=f"{name.lower()}::concept",
                    entity_type="concept",
                )
            )
    db_session.commit()
    relation_id = relation.id
    candidate_id = candidate.id

    delete_document_graph_incremental(db_session, sample_course.id, document_a.id)
    db_session.commit()
    db_session.expire_all()

    refreshed_relation = db_session.get(ConceptRelation, relation_id)
    refreshed_candidate = db_session.get(GraphRelationCandidate, candidate_id)
    assert refreshed_relation.source_document_ids == [document_b.id]
    assert refreshed_relation.evidence_chunk_id is None
    assert refreshed_relation.support_count == 2
    assert refreshed_candidate.source_document_ids == [document_b.id]
    assert refreshed_candidate.evidence_chunk_id is None
    assert refreshed_candidate.support_count == 2


def test_cleanup_stale_data_rejects_active_batch(db_session, sample_course):
    from app.models import IngestionBatch
    from app.services.maintenance import MaintenanceConflict, cleanup_stale_data

    db_session.add(IngestionBatch(course_id=sample_course.id, source_root="unit", trigger_source="upload", status="queued"))
    db_session.commit()

    with pytest.raises(MaintenanceConflict):
        cleanup_stale_data(db_session, sample_course.id, sample_course.name)


@pytest.mark.asyncio
async def test_rebuild_graph_endpoint_accepts_mode_parameter(db_session, sample_course, monkeypatch):
    from fastapi import BackgroundTasks

    from app.api import rebuild_graph_endpoint
    from app.services.ingestion import run_graph_rebuild_background
    from app.core.config import get_settings
    from app.schemas import RebuildGraphRequest

    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    monkeypatch.setenv("INGESTION_EXECUTION_MODE", "inline")
    get_settings.cache_clear()
    background_tasks = BackgroundTasks()

    response = await rebuild_graph_endpoint(
        background_tasks,
        request=RebuildGraphRequest(mode="incremental"),
        course_id=sample_course.id,
        db=db_session,
    )

    assert response["mode"] == "incremental"
    assert response["batch_id"]
    assert response["state"] == "extracting_graph"
    assert background_tasks.tasks
    assert background_tasks.tasks[0].func is run_graph_rebuild_background


@pytest.mark.asyncio
async def test_rebuild_graph_endpoint_dry_run_has_no_batch(db_session, sample_course, monkeypatch):
    from fastapi import BackgroundTasks

    from app.api import rebuild_graph_endpoint
    from app.core.config import get_settings
    from app.schemas import RebuildGraphRequest

    monkeypatch.setenv("ENABLE_MODEL_FALLBACK", "false")
    get_settings.cache_clear()

    response = await rebuild_graph_endpoint(
        BackgroundTasks(),
        request=RebuildGraphRequest(mode="full", dry_run=True),
        course_id=sample_course.id,
        db=db_session,
    )

    assert response["batch_id"] is None
    assert response["state"] == "dry_run"
    assert response["mode"] == "full"
    assert response["dry_run"] is True
    assert response["affected_documents"] >= 0


def test_delete_course_data_removes_database_vectors_and_directory(db_session, sample_course, monkeypatch):
    from app.core.config import get_settings
    from app.models import (
        AgentRun,
        AgentTraceEvent,
        ConceptRelation,
        CourseModelHyperparameter,
        Course,
        EntityMention,
        EntityMergeCandidate,
        GraphCommunitySummary,
        GraphExtractionChunkTask,
        GraphExtractionRun,
        GraphHpoJudgeSample,
        GraphHpoObjectiveModel,
        GraphRelationCandidate,
        IngestionBatch,
        IngestionCompensationLog,
        IngestionJob,
        IngestionLog,
        QualityProfile,
        QASession,
    )
    from app.services import maintenance
    from app.services.maintenance import delete_course_data

    document, version, chunk = add_document_graph(db_session, sample_course.id, active=True, title="DeleteMe")
    source = add_concept(db_session, sample_course.id, "Delete Source")
    target = add_concept(db_session, sample_course.id, "Delete Target")
    relation = ConceptRelation(
        course_id=sample_course.id,
        source_concept_id=source.id,
        target_concept_id=target.id,
        target_name=target.canonical_name,
        relation_type="related_to",
        evidence_chunk_id=chunk.id,
    )
    batch = IngestionBatch(course_id=sample_course.id, source_root="unit", trigger_source="upload", status="completed")
    job = IngestionJob(course_id=sample_course.id, document_id=document.id, source_path=document.source_path, status="completed")
    session = QASession(course_id=sample_course.id, title="chat", transcript=[])
    run = AgentRun(course_id=sample_course.id, question="q", status="completed")
    db_session.add(batch)
    db_session.flush()
    graph_run = GraphExtractionRun(course_id=sample_course.id, batch_id=batch.id, strategy="adaptive_best_first", status="completed")
    db_session.add_all([relation, job, session, run, graph_run])
    db_session.flush()
    db_session.add_all(
        [
            IngestionLog(batch_id=batch.id, event="done", message="done"),
            IngestionCompensationLog(course_id=sample_course.id, job_id=job.id, operation="delete", vector_ids=[chunk.id], status="completed"),
            AgentTraceEvent(run_id=run.id, node="n", status="completed", document_ids=[], scores={}),
            GraphExtractionChunkTask(
                run_id=graph_run.id,
                course_id=sample_course.id,
                chunk_id=chunk.id,
                chunk_hash="hash",
                status="completed",
                payload_json={"concepts": []},
            ),
            CourseModelHyperparameter(
                course_id=sample_course.id,
                llm_model_name="judge-llm",
                embedding_model_name="embedder",
                embedding_text_version="metadata_enriched_v1",
                model_name="llm:judge-llm|embedding:embedder|text:metadata_enriched_v1",
                hpo_status="completed",
            ),
            GraphHpoJudgeSample(
                course_id=sample_course.id,
                llm_model_name="judge-llm",
                embedding_model_name="embedder",
                embedding_text_version="metadata_enriched_v1",
                model_key="llm:judge-llm|embedding:embedder|text:metadata_enriched_v1",
                payload_fingerprint="fingerprint",
                prompt_version="graph_hpo_judge_v1",
                candidate_a_params={},
                candidate_b_params={},
                candidate_a_features={},
                candidate_b_features={},
                winner="a",
                confidence=0.8,
            ),
            GraphHpoObjectiveModel(
                course_id=sample_course.id,
                llm_model_name="judge-llm",
                embedding_model_name="embedder",
                embedding_text_version="metadata_enriched_v1",
                model_key="llm:judge-llm|embedding:embedder|text:metadata_enriched_v1",
                objective_version="graph_hpo_judge_learned_v1",
                payload_fingerprint="fingerprint",
                feature_names=["edge_evidence_coverage"],
                weights_json={"edge_evidence_coverage": 1.0},
                normalization_json={},
                label_count=6,
            ),
            QualityProfile(course_id=sample_course.id, version="v1", profile_json={}, sample_chunk_ids=[]),
            GraphCommunitySummary(course_id=sample_course.id, algorithm="louvain", community_id=1, version="v1"),
            GraphRelationCandidate(
                course_id=sample_course.id,
                source_concept_id=source.id,
                target_concept_id=target.id,
                target_name=target.canonical_name,
                relation_type="related_to",
                evidence_chunk_id=chunk.id,
            ),
            EntityMention(
                course_id=sample_course.id,
                chunk_id=chunk.id,
                document_id=document.id,
                concept_id=source.id,
                surface="Delete Source",
                canonical_name="Delete Source",
                normalized_key="delete source::concept",
                entity_type="concept",
            ),
            EntityMergeCandidate(
                course_id=sample_course.id,
                left_key="delete source::concept",
                right_key="delete target::concept",
                entity_type="concept",
            ),
        ]
    )
    db_session.commit()

    course_root = get_settings().course_paths_for_name(sample_course.name)["course_root"]
    (course_root / "storage").mkdir(parents=True, exist_ok=True)
    (course_root / "storage" / "note.md").write_text("delete me", encoding="utf-8")
    deleted_vectors: list[str] = []
    operations: list[str] = []

    class TrackingVectorStore:
        def __init__(self, course_name=None):
            self.course_name = course_name

        def list_ids(self, course_id=None):
            return [chunk.id, "orphan-vector"]

        def delete(self, ids):
            operations.append("qdrant_delete")
            deleted_vectors.extend(ids)

    monkeypatch.setattr(maintenance, "VectorStore", TrackingVectorStore)
    original_commit = db_session.commit

    def tracking_commit():
        operations.append("db_commit")
        original_commit()

    monkeypatch.setattr(db_session, "commit", tracking_commit)
    course_id = sample_course.id
    chunk_id = chunk.id

    stats = delete_course_data(db_session, sample_course)

    assert stats["deleted_vectors"] == 2
    assert set(deleted_vectors) == {chunk_id, "orphan-vector"}
    assert stats["deleted_graph_extraction_tasks"] == 1
    assert stats["deleted_graph_extraction_runs"] == 1
    assert stats["deleted_hpo_judge_samples"] == 1
    assert stats["deleted_hpo_objective_models"] == 1
    assert stats["deleted_model_hyperparameters"] == 1
    assert stats["deleted_quality_profiles"] == 1
    assert stats["deleted_community_summaries"] == 1
    assert stats["deleted_relation_candidates"] == 1
    assert stats["deleted_mentions"] == 1
    assert stats["deleted_merge_candidates"] == 1
    assert stats["deleted_courses"] == 1
    assert stats["deleted_directory"] == 1
    assert operations.index("db_commit") < operations.index("qdrant_delete")
    assert not course_root.exists()
    assert db_session.get(Course, course_id) is None


def test_cleanup_stale_data_commits_db_before_qdrant_delete(db_session, sample_course, monkeypatch):
    """Regression: DB commit must happen before Qdrant vector deletion to maintain cross-store consistency."""
    from app.services import maintenance
    from app.services.maintenance import cleanup_stale_data

    operations: list[str] = []

    class TrackingVectorStore:
        def __init__(self, course_name=None):
            self.course_name = course_name

        def list_ids(self, course_id=None):
            return ["stale-vector-1"]

        def delete(self, ids):
            operations.append("qdrant_delete")

    original_commit = db_session.commit

    def tracking_commit():
        operations.append("db_commit")
        original_commit()

    monkeypatch.setattr(maintenance, "VectorStore", TrackingVectorStore)
    monkeypatch.setattr(db_session, "commit", tracking_commit)

    cleanup_stale_data(db_session, sample_course.id, sample_course.name)

    db_commit_indices = [i for i, op in enumerate(operations) if op == "db_commit"]
    qdrant_index = operations.index("qdrant_delete")
    assert any(idx < qdrant_index for idx in db_commit_indices), (
        f"DB commit must occur before Qdrant delete, but operations were: {operations}"
    )

from __future__ import annotations

import shutil
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    AgentRun,
    AgentTraceEvent,
    Chunk,
    Concept,
    ConceptAlias,
    ConceptRelation,
    Course,
    CourseModelHyperparameter,
    Document,
    DocumentVersion,
    EntityMergeCandidate,
    EntityMention,
    GraphExtractionChunkTask,
    GraphExtractionRun,
    GraphCommunitySummary,
    GraphHpoJudgeSample,
    GraphHpoObjectiveModel,
    GraphRelationCandidate,
    IngestionBatch,
    IngestionCompensationLog,
    IngestionJob,
    IngestionLog,
    QASession,
    QualityProfile,
)
from app.services.concept_graph import is_valid_concept, normalize_relation_type
from app.services.ingestion import active_batch_for_course
from app.services.vector_store import VectorStore


class MaintenanceConflict(RuntimeError):
    pass


@dataclass
class GraphCleanupStats:
    removed_relations: int = 0
    removed_aliases: int = 0
    removed_concepts: int = 0
    migrated_relations: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "removed_relations": self.removed_relations,
            "removed_aliases": self.removed_aliases,
            "removed_concepts": self.removed_concepts,
            "migrated_relations": self.migrated_relations,
        }


def ensure_no_active_batch(db: Session, course_id: str) -> None:
    if active_batch_for_course(db, course_id) is not None:
        raise MaintenanceConflict("Cannot run maintenance while an ingestion batch is active")


def cleanup_stale_graph_references(db: Session, course_id: str) -> GraphCleanupStats:
    relations = db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all()
    if not relations:
        return GraphCleanupStats()

    chunk_ids = {relation.evidence_chunk_id for relation in relations if relation.evidence_chunk_id}
    chunks = {chunk.id: chunk for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(chunk_ids))).all()} if chunk_ids else {}
    document_ids = {chunk.document_id for chunk in chunks.values()}
    documents = {document.id: document for document in db.scalars(select(Document).where(Document.id.in_(document_ids))).all()} if document_ids else {}

    concept_ids = {
        concept_id
        for relation in relations
        for concept_id in (relation.source_concept_id, relation.target_concept_id)
        if concept_id
    }
    concepts = {concept.id: concept for concept in db.scalars(select(Concept).where(Concept.id.in_(concept_ids))).all()} if concept_ids else {}

    stale_relation_ids: list[str] = []
    affected_concept_ids: set[str] = set()
    migrated_relations = 0
    for relation in relations:
        stale = False
        relation_type = normalize_relation_type(relation.relation_type)
        if not relation_type:
            stale = True
        elif relation_type != relation.relation_type:
            relation.relation_type = relation_type
            metadata = dict(getattr(relation, "metadata_json", None) or {})
            metadata["legacy_relation_type_migrated"] = True
            relation.metadata_json = metadata
            stats_migrated = True
            migrated_relations += 1
        else:
            stats_migrated = False
        source_concept = concepts.get(relation.source_concept_id)
        target_concept = concepts.get(relation.target_concept_id or "")
        if source_concept is None or not is_valid_concept(source_concept.canonical_name):
            stale = True
        if relation.target_concept_id and (target_concept is None or not is_valid_concept(target_concept.canonical_name)):
            stale = True
        if relation.evidence_chunk_id:
            chunk = chunks.get(relation.evidence_chunk_id)
            document = documents.get(chunk.document_id) if chunk else None
            if chunk is None or not chunk.is_active or document is None or not document.is_active:
                stale = True
        if stale:
            stale_relation_ids.append(relation.id)
            affected_concept_ids.update(
                concept_id
                for concept_id in (relation.source_concept_id, relation.target_concept_id)
                if concept_id
            )
        elif stats_migrated:
            affected_concept_ids.update(
                concept_id
                for concept_id in (relation.source_concept_id, relation.target_concept_id)
                if concept_id
            )

    stats = GraphCleanupStats()
    stats.migrated_relations = migrated_relations
    if stale_relation_ids:
        stats.removed_relations = db.query(ConceptRelation).filter(ConceptRelation.id.in_(stale_relation_ids)).delete(synchronize_session="fetch")

    if affected_concept_ids:
        remaining_relations = db.scalars(
            select(ConceptRelation).where(
                ConceptRelation.course_id == course_id,
                (
                    ConceptRelation.source_concept_id.in_(affected_concept_ids)
                    | ConceptRelation.target_concept_id.in_(affected_concept_ids)
                ),
            )
        ).all()
        still_referenced = {
            concept_id
            for relation in remaining_relations
            for concept_id in (relation.source_concept_id, relation.target_concept_id)
            if concept_id
        }
        orphan_concept_ids = sorted(affected_concept_ids - still_referenced)
        if orphan_concept_ids:
            db.query(GraphRelationCandidate).filter(
                or_(
                    GraphRelationCandidate.source_concept_id.in_(orphan_concept_ids),
                    GraphRelationCandidate.target_concept_id.in_(orphan_concept_ids),
                )
            ).delete(synchronize_session="fetch")
            stats.removed_aliases = db.query(ConceptAlias).filter(ConceptAlias.concept_id.in_(orphan_concept_ids)).delete(synchronize_session="fetch")
            stats.removed_concepts = db.query(Concept).filter(
                Concept.course_id == course_id,
                Concept.id.in_(orphan_concept_ids),
            ).delete(synchronize_session="fetch")

    return stats


def cleanup_stale_graph(db: Session, course_id: str) -> dict[str, int]:
    ensure_no_active_batch(db, course_id)
    stats = cleanup_stale_graph_references(db, course_id)
    db.commit()
    return stats.as_dict()


def _metadata_chunk_ids(metadata: dict | None) -> set[str]:
    values: set[str] = set()
    if not isinstance(metadata, dict):
        return values
    for key in ("chunk_ids", "evidence_chunk_ids", "support_chunk_ids"):
        raw = metadata.get(key)
        if isinstance(raw, str):
            values.add(raw)
        elif isinstance(raw, list):
            values.update(str(item) for item in raw if item)
    return values


def _active_chunk_ids(db: Session, course_id: str, chunk_ids: set[str]) -> set[str]:
    if not chunk_ids:
        return set()
    return set(
        db.scalars(
            select(Chunk.id).where(
                Chunk.course_id == course_id,
                Chunk.id.in_(chunk_ids),
                Chunk.is_active.is_(True),
            )
        ).all()
    )


def _concept_mention_chunk_ids(
    db: Session,
    course_id: str,
    concept_id: str | None,
    source_document_ids: set[str],
) -> set[str]:
    if not concept_id:
        return set()
    query = select(EntityMention.chunk_id).where(
        EntityMention.course_id == course_id,
        EntityMention.concept_id == concept_id,
        EntityMention.chunk_id.is_not(None),
    )
    if source_document_ids:
        query = query.where(EntityMention.document_id.in_(source_document_ids))
    return {str(chunk_id) for chunk_id in db.scalars(query).all() if chunk_id}


def _recalculate_edge_support_count(
    db: Session,
    *,
    course_id: str,
    source_concept_id: str,
    target_concept_id: str | None,
    source_document_ids: list[str] | None,
    evidence_chunk_id: str | None,
    metadata_json: dict | None,
) -> int:
    source_ids = {str(item) for item in (source_document_ids or []) if item}
    evidence_chunk_ids = _metadata_chunk_ids(metadata_json)
    if evidence_chunk_id:
        evidence_chunk_ids.add(str(evidence_chunk_id))

    source_chunks = _concept_mention_chunk_ids(db, course_id, source_concept_id, source_ids)
    if target_concept_id:
        target_chunks = _concept_mention_chunk_ids(db, course_id, target_concept_id, source_ids)
        co_mention_chunks = source_chunks.intersection(target_chunks)
        evidence_chunk_ids.update(co_mention_chunks)
    else:
        evidence_chunk_ids.update(source_chunks)

    active_evidence_chunks = _active_chunk_ids(db, course_id, evidence_chunk_ids)
    if active_evidence_chunks:
        return len(active_evidence_chunks)
    return max(1, len(source_ids))


def delete_document_graph_incremental(db: Session, course_id: str, document_id: str) -> dict[str, int]:
    """Remove one document's graph evidence without committing.

    Concepts and relations shared with other documents are retained with the
    document_id stripped from their source lists. Orphan concepts and dangling
    relation/candidate rows are physically removed.
    """
    document_ids = {document_id}
    chunk_ids = set(db.scalars(select(Chunk.id).where(Chunk.course_id == course_id, Chunk.document_id == document_id)).all())
    affected_concept_ids = set(
        db.scalars(
            select(EntityMention.concept_id).where(
                EntityMention.course_id == course_id,
                EntityMention.document_id == document_id,
                EntityMention.concept_id.is_not(None),
            )
        ).all()
    )
    affected_concept_ids.discard(None)
    removed_mentions = db.query(EntityMention).filter(
        EntityMention.course_id == course_id,
        EntityMention.document_id == document_id,
    ).delete(synchronize_session=False)

    removed_relations = 0
    for relation in db.scalars(select(ConceptRelation).where(ConceptRelation.course_id == course_id)).all():
        sources = set(relation.source_document_ids or [])
        evidence_from_document = bool(relation.evidence_chunk_id and relation.evidence_chunk_id in chunk_ids)
        affected_concept_ids.add(relation.source_concept_id)
        if relation.target_concept_id:
            affected_concept_ids.add(relation.target_concept_id)
        if not sources and evidence_from_document:
            db.delete(relation)
            removed_relations += 1
            continue
        if sources and sources.issubset(document_ids):
            db.delete(relation)
            removed_relations += 1
            continue
        if sources.intersection(document_ids):
            relation.source_document_ids = sorted(sources - document_ids)
            if evidence_from_document:
                relation.evidence_chunk_id = None
            relation.support_count = _recalculate_edge_support_count(
                db,
                course_id=course_id,
                source_concept_id=relation.source_concept_id,
                target_concept_id=relation.target_concept_id,
                source_document_ids=relation.source_document_ids,
                evidence_chunk_id=relation.evidence_chunk_id,
                metadata_json=relation.metadata_json,
            )

    removed_candidates = 0
    for candidate in db.scalars(select(GraphRelationCandidate).where(GraphRelationCandidate.course_id == course_id)).all():
        sources = set(candidate.source_document_ids or [])
        evidence_from_document = bool(candidate.evidence_chunk_id and candidate.evidence_chunk_id in chunk_ids)
        affected_concept_ids.add(candidate.source_concept_id)
        if candidate.target_concept_id:
            affected_concept_ids.add(candidate.target_concept_id)
        if (not sources and evidence_from_document) or (sources and sources.issubset(document_ids)):
            db.delete(candidate)
            removed_candidates += 1
        elif sources.intersection(document_ids):
            candidate.source_document_ids = sorted(sources - document_ids)
            if evidence_from_document:
                candidate.evidence_chunk_id = None
            candidate.support_count = _recalculate_edge_support_count(
                db,
                course_id=course_id,
                source_concept_id=candidate.source_concept_id,
                target_concept_id=candidate.target_concept_id,
                source_document_ids=candidate.source_document_ids,
                evidence_chunk_id=candidate.evidence_chunk_id,
                metadata_json=candidate.metadata_json,
            )

    removed_concepts = 0
    removed_aliases = 0
    concepts_to_delete: set[str] = set()
    for concept in db.scalars(select(Concept).where(Concept.course_id == course_id)).all():
        sources = set(concept.source_document_ids or [])
        if sources.intersection(document_ids):
            affected_concept_ids.add(concept.id)
            sources -= document_ids
        if concept.id in affected_concept_ids:
            mention_rows = db.execute(
                select(EntityMention.document_id, EntityMention.chunk_id).where(
                    EntityMention.course_id == course_id,
                    EntityMention.concept_id == concept.id,
                )
            ).all()
            mention_doc_ids = {row[0] for row in mention_rows if row[0]}
            mention_chunk_ids = {row[1] for row in mention_rows if row[1]}
            sources.update(mention_doc_ids)
            concept.source_document_ids = sorted(sources)
            concept.evidence_count = max(len(mention_rows), len(mention_chunk_ids), len(sources))

    db.flush()
    incident_relation_ids = set(
        db.scalars(
            select(ConceptRelation.source_concept_id).where(ConceptRelation.course_id == course_id)
        ).all()
    )
    incident_relation_ids.update(
        concept_id
        for concept_id in db.scalars(
            select(ConceptRelation.target_concept_id).where(
                ConceptRelation.course_id == course_id,
                ConceptRelation.target_concept_id.is_not(None),
            )
        ).all()
        if concept_id
    )
    incident_relation_ids.update(
        db.scalars(
            select(GraphRelationCandidate.source_concept_id).where(GraphRelationCandidate.course_id == course_id)
        ).all()
    )
    incident_relation_ids.update(
        concept_id
        for concept_id in db.scalars(
            select(GraphRelationCandidate.target_concept_id).where(
                GraphRelationCandidate.course_id == course_id,
                GraphRelationCandidate.target_concept_id.is_not(None),
            )
        ).all()
        if concept_id
    )
    for concept in db.scalars(select(Concept).where(Concept.course_id == course_id)).all():
        if concept.source_document_ids or concept.evidence_count > 0:
            continue
        if concept.id in incident_relation_ids:
            continue
        has_mentions = db.scalar(
            select(EntityMention.id).where(EntityMention.course_id == course_id, EntityMention.concept_id == concept.id).limit(1)
        )
        if has_mentions:
            continue
        concepts_to_delete.add(concept.id)

    if concepts_to_delete:
        deleted_concepts = db.scalars(select(Concept).where(Concept.id.in_(concepts_to_delete))).all()
        deleted_keys = {concept.normalized_name for concept in deleted_concepts if concept.normalized_name}
        removed_relations += db.query(ConceptRelation).filter(
            ConceptRelation.course_id == course_id,
            (
                ConceptRelation.source_concept_id.in_(concepts_to_delete)
                | ConceptRelation.target_concept_id.in_(concepts_to_delete)
            ),
        ).delete(synchronize_session=False)
        removed_candidates += db.query(GraphRelationCandidate).filter(
            GraphRelationCandidate.course_id == course_id,
            (
                GraphRelationCandidate.source_concept_id.in_(concepts_to_delete)
                | GraphRelationCandidate.target_concept_id.in_(concepts_to_delete)
            ),
        ).delete(synchronize_session=False)
        removed_aliases = db.query(ConceptAlias).filter(ConceptAlias.concept_id.in_(concepts_to_delete)).delete(synchronize_session=False)
        if deleted_keys:
            db.query(EntityMergeCandidate).filter(
                EntityMergeCandidate.course_id == course_id,
                (
                    EntityMergeCandidate.left_key.in_(deleted_keys)
                    | EntityMergeCandidate.right_key.in_(deleted_keys)
                ),
            ).delete(synchronize_session=False)
        removed_concepts = db.query(Concept).filter(
            Concept.course_id == course_id,
            Concept.id.in_(concepts_to_delete),
        ).delete(synchronize_session=False)

    db.flush()
    return {
        "graph_removed_mentions": removed_mentions,
        "graph_removed_relations": removed_relations,
        "graph_removed_candidates": removed_candidates,
        "graph_removed_aliases": removed_aliases,
        "graph_removed_concepts": removed_concepts,
    }


def cleanup_stale_data(db: Session, course_id: str, course_name: str) -> dict[str, int]:
    from app.services.ingestion import create_vector_compensation_log, mark_vector_compensation_log

    ensure_no_active_batch(db, course_id)
    graph_stats = cleanup_stale_graph_references(db, course_id)

    active_chunk_ids = set(
        db.scalars(select(Chunk.id).where(Chunk.course_id == course_id, Chunk.is_active.is_(True))).all()
    )
    vector_store = VectorStore(course_name=course_name)
    stale_vector_ids = sorted(set(vector_store.list_ids(course_id)) - active_chunk_ids)

    inactive_document_ids = set(
        db.scalars(select(Document.id).where(Document.course_id == course_id, Document.is_active.is_(False))).all()
    )
    if inactive_document_ids:
        db.query(IngestionJob).filter(IngestionJob.document_id.in_(inactive_document_ids)).update(
            {"document_id": None},
            synchronize_session="fetch",
        )

    inactive_version_ids = db.scalars(
        select(DocumentVersion.id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            Document.course_id == course_id,
            or_(DocumentVersion.is_active.is_(False), Document.is_active.is_(False)),
        )
    ).all()
    inactive_chunk_ids = set(
        db.scalars(select(Chunk.id).where(Chunk.course_id == course_id, Chunk.is_active.is_(False))).all()
    )
    if inactive_chunk_ids:
        db.query(GraphExtractionChunkTask).filter(
            GraphExtractionChunkTask.chunk_id.in_(inactive_chunk_ids)
        ).delete(synchronize_session="fetch")
    deleted_chunks = db.query(Chunk).filter(Chunk.course_id == course_id, Chunk.is_active.is_(False)).delete(synchronize_session="fetch")
    deleted_document_versions = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.id.in_(inactive_version_ids))
        .delete(synchronize_session="fetch")
        if inactive_version_ids
        else 0
    )
    deleted_documents = db.query(Document).filter(Document.course_id == course_id, Document.is_active.is_(False)).delete(synchronize_session="fetch")
    db.commit()

    # Delete Qdrant vectors AFTER DB commit to maintain cross-store consistency.
    if stale_vector_ids:
        delete_log = create_vector_compensation_log(
            db,
            course_id=course_id,
            job_id=None,
            operation="delete",
            vector_ids=stale_vector_ids,
            payload_json={"source": "cleanup_stale_data"},
        )
        try:
            vector_store.delete(stale_vector_ids)
            mark_vector_compensation_log(db, delete_log.id, "completed")
        except Exception as exc:
            mark_vector_compensation_log(db, delete_log.id, "failed", str(exc))
            raise

    return {
        "deleted_vectors": len(stale_vector_ids),
        "deleted_chunks": deleted_chunks,
        "deleted_document_versions": deleted_document_versions,
        "deleted_documents": deleted_documents,
        "removed_graph_relations": graph_stats.removed_relations,
        "removed_graph_concepts": graph_stats.removed_concepts,
    }


def delete_course_data(db: Session, course: Course) -> dict[str, int]:
    ensure_no_active_batch(db, course.id)
    settings = get_settings()
    course_paths = settings.course_paths_for_name(course.name)
    data_root = settings.data_root.resolve()
    course_root = course_paths["course_root"].resolve()
    if course_root != data_root and data_root not in course_root.parents:
        raise RuntimeError(f"Refusing to delete course directory outside DATA_ROOT: {course_root}")

    vector_store = VectorStore(course_name=course.name)
    vector_ids = vector_store.list_ids(course.id)

    run_ids = db.scalars(select(AgentRun.id).where(AgentRun.course_id == course.id)).all()
    batch_ids = db.scalars(select(IngestionBatch.id).where(IngestionBatch.course_id == course.id)).all()
    concept_ids = db.scalars(select(Concept.id).where(Concept.course_id == course.id)).all()
    document_ids = db.scalars(select(Document.id).where(Document.course_id == course.id)).all()
    graph_run_ids = db.scalars(select(GraphExtractionRun.id).where(GraphExtractionRun.course_id == course.id)).all()

    deleted_trace_events = db.query(AgentTraceEvent).filter(AgentTraceEvent.run_id.in_(run_ids)).delete(synchronize_session=False) if run_ids else 0
    deleted_agent_runs = db.query(AgentRun).filter(AgentRun.course_id == course.id).delete(synchronize_session=False)
    deleted_sessions = db.query(QASession).filter(QASession.course_id == course.id).delete(synchronize_session=False)

    deleted_ingestion_logs = db.query(IngestionLog).filter(IngestionLog.batch_id.in_(batch_ids)).delete(synchronize_session=False) if batch_ids else 0
    deleted_compensations = db.query(IngestionCompensationLog).filter(IngestionCompensationLog.course_id == course.id).delete(synchronize_session=False)
    deleted_jobs = db.query(IngestionJob).filter(IngestionJob.course_id == course.id).delete(synchronize_session=False)
    deleted_graph_extraction_tasks = db.query(GraphExtractionChunkTask).filter(GraphExtractionChunkTask.course_id == course.id).delete(synchronize_session=False)
    deleted_graph_extraction_runs = db.query(GraphExtractionRun).filter(GraphExtractionRun.id.in_(graph_run_ids)).delete(synchronize_session=False) if graph_run_ids else 0
    deleted_batches = db.query(IngestionBatch).filter(IngestionBatch.course_id == course.id).delete(synchronize_session=False)

    deleted_hpo_judge_samples = db.query(GraphHpoJudgeSample).filter(GraphHpoJudgeSample.course_id == course.id).delete(synchronize_session=False)
    deleted_hpo_objective_models = db.query(GraphHpoObjectiveModel).filter(GraphHpoObjectiveModel.course_id == course.id).delete(synchronize_session=False)
    deleted_model_hyperparameters = db.query(CourseModelHyperparameter).filter(CourseModelHyperparameter.course_id == course.id).delete(synchronize_session=False)
    deleted_quality_profiles = db.query(QualityProfile).filter(QualityProfile.course_id == course.id).delete(synchronize_session=False)
    deleted_community_summaries = db.query(GraphCommunitySummary).filter(GraphCommunitySummary.course_id == course.id).delete(synchronize_session=False)
    deleted_relation_candidates = db.query(GraphRelationCandidate).filter(GraphRelationCandidate.course_id == course.id).delete(synchronize_session=False)
    deleted_mentions = db.query(EntityMention).filter(EntityMention.course_id == course.id).delete(synchronize_session=False)
    deleted_merge_candidates = db.query(EntityMergeCandidate).filter(EntityMergeCandidate.course_id == course.id).delete(synchronize_session=False)
    deleted_relations = db.query(ConceptRelation).filter(ConceptRelation.course_id == course.id).delete(synchronize_session=False)
    deleted_aliases = db.query(ConceptAlias).filter(ConceptAlias.concept_id.in_(concept_ids)).delete(synchronize_session=False) if concept_ids else 0
    deleted_concepts = db.query(Concept).filter(Concept.course_id == course.id).delete(synchronize_session=False)

    deleted_chunks = db.query(Chunk).filter(Chunk.course_id == course.id).delete(synchronize_session=False)
    deleted_versions = db.query(DocumentVersion).filter(DocumentVersion.document_id.in_(document_ids)).delete(synchronize_session=False) if document_ids else 0
    deleted_documents = db.query(Document).filter(Document.course_id == course.id).delete(synchronize_session=False)

    db.delete(course)
    db.commit()

    vector_store.delete(vector_ids)

    deleted_directory = 0
    if course_root.exists():
        shutil.rmtree(course_root)
        deleted_directory = 1

    return {
        "deleted_vectors": len(vector_ids),
        "deleted_trace_events": deleted_trace_events,
        "deleted_agent_runs": deleted_agent_runs,
        "deleted_sessions": deleted_sessions,
        "deleted_ingestion_logs": deleted_ingestion_logs,
        "deleted_compensations": deleted_compensations,
        "deleted_jobs": deleted_jobs,
        "deleted_graph_extraction_tasks": deleted_graph_extraction_tasks,
        "deleted_graph_extraction_runs": deleted_graph_extraction_runs,
        "deleted_batches": deleted_batches,
        "deleted_hpo_judge_samples": deleted_hpo_judge_samples,
        "deleted_hpo_objective_models": deleted_hpo_objective_models,
        "deleted_model_hyperparameters": deleted_model_hyperparameters,
        "deleted_quality_profiles": deleted_quality_profiles,
        "deleted_community_summaries": deleted_community_summaries,
        "deleted_relation_candidates": deleted_relation_candidates,
        "deleted_mentions": deleted_mentions,
        "deleted_merge_candidates": deleted_merge_candidates,
        "deleted_relations": deleted_relations,
        "deleted_aliases": deleted_aliases,
        "deleted_concepts": deleted_concepts,
        "deleted_chunks": deleted_chunks,
        "deleted_document_versions": deleted_versions,
        "deleted_documents": deleted_documents,
        "deleted_courses": 1,
        "deleted_directory": deleted_directory,
    }

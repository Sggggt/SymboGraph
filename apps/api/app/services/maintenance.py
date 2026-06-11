from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    ActiveChunk,
    AgentRun,
    AgentTraceEvent,
    AnswerSession,
    CitationVerification,
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
    IngestionBatch,
    IngestionCompensationLog,
    IngestionJob,
    IngestionLog,
    KnowledgeBase,
    ParseJob,
    PolicyObservation,
    PolicyState,
    ProjectionCommunity,
    ProjectionEdge,
    ProjectionNode,
    ProjectionState,
    QASession,
    QualityDecision,
    QualityObservation,
    RetrievalTrace,
    RewardEvent,
    SignalCandidate,
    SignalCommunity,
    SignalCommunityMembership,
    SignalDecision,
    SignalEdge,
    SignalNode,
    SignalRelationSpec,
    SignalSchemaState,
    SignalState,
    SignalTypeSpec,
    SourceFile,
    VectorRecord,
)
from app.services.ingestion import active_batch_for_knowledge_base
from app.services.vector_store import VectorStore


class MaintenanceConflict(RuntimeError):
    pass


@dataclass
class EvidenceCleanupStats:
    removed_evidence_atoms: int = 0
    removed_evidence_edges: int = 0
    removed_evidence_graph_states: int = 0
    removed_chunk_candidates: int = 0
    removed_chunk_decisions: int = 0
    removed_quality_decisions: int = 0
    removed_active_chunks: int = 0
    removed_community_states: int = 0
    removed_community_memberships: int = 0
    removed_community_summaries: int = 0
    removed_signal_schema_states: int = 0
    removed_signal_states: int = 0
    removed_signal_candidates: int = 0
    removed_signal_decisions: int = 0
    removed_signal_nodes: int = 0
    removed_signal_edges: int = 0
    removed_signal_communities: int = 0
    removed_signal_community_memberships: int = 0
    removed_projection_states: int = 0
    removed_projection_nodes: int = 0
    removed_projection_edges: int = 0
    removed_projection_communities: int = 0
    removed_vector_records: int = 0

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def ensure_no_active_batch(db: Session, knowledge_base_id: str) -> None:
    if active_batch_for_knowledge_base(db, knowledge_base_id) is not None:
        raise MaintenanceConflict("Cannot run maintenance while an ingestion batch is active")


def _json_ids(raw: object) -> set[str]:
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, list):
        return {str(item) for item in raw if item}
    return set()


def _active_chunk_ids_for_document(db: Session, knowledge_base_id: str, document_id: str) -> set[str]:
    ids: set[str] = set()
    for chunk in db.scalars(select(ActiveChunk).where(ActiveChunk.knowledge_base_id == knowledge_base_id)).all():
        metadata = chunk.metadata_json or {}
        if metadata.get("document_id") == document_id:
            ids.add(chunk.id)
    return ids


def _active_chunk_ids_touching_atoms(db: Session, knowledge_base_id: str, atom_ids: set[str]) -> set[str]:
    if not atom_ids:
        return set()
    ids: set[str] = set()
    for chunk in db.scalars(select(ActiveChunk).where(ActiveChunk.knowledge_base_id == knowledge_base_id)).all():
        if _json_ids(chunk.atom_ids_json) & atom_ids:
            ids.add(chunk.id)
    return ids


def _graph_state_ids_touching_atoms(
    db: Session,
    *,
    knowledge_base_id: str,
    stale_atom_ids: set[str],
    inactive_document_version_ids: set[str],
) -> set[str]:
    stale_graph_state_ids: set[str] = set()
    active_atom_ids = set(
        db.scalars(
            select(EvidenceAtom.id).where(
                EvidenceAtom.knowledge_base_id == knowledge_base_id,
                EvidenceAtom.state == "active",
            )
        ).all()
    ) - stale_atom_ids
    for state in db.scalars(select(EvidenceGraphState).where(EvidenceGraphState.knowledge_base_id == knowledge_base_id)).all():
        state_version_ids = _json_ids(state.active_document_version_ids)
        state_atom_ids = _json_ids(state.active_atom_ids)
        if (
            state.state != "active"
            or bool(state_version_ids & inactive_document_version_ids)
            or bool(state_atom_ids & stale_atom_ids)
            or not state_atom_ids.issubset(active_atom_ids)
        ):
            stale_graph_state_ids.add(state.id)
    return stale_graph_state_ids


def _delete_signal_and_projection_state(db: Session, stats: EvidenceCleanupStats, signal_state_ids: set[str]) -> None:
    if not signal_state_ids:
        return
    projection_state_ids = set(
        db.scalars(select(ProjectionState.id).where(ProjectionState.signal_state_id.in_(signal_state_ids))).all()
    )
    if projection_state_ids:
        stats.removed_projection_communities = db.query(ProjectionCommunity).filter(
            ProjectionCommunity.projection_state_id.in_(projection_state_ids)
        ).delete(synchronize_session=False)
        stats.removed_projection_edges = db.query(ProjectionEdge).filter(
            ProjectionEdge.projection_state_id.in_(projection_state_ids)
        ).delete(synchronize_session=False)
        stats.removed_projection_nodes = db.query(ProjectionNode).filter(
            ProjectionNode.projection_state_id.in_(projection_state_ids)
        ).delete(synchronize_session=False)
        stats.removed_projection_states = db.query(ProjectionState).filter(
            ProjectionState.id.in_(projection_state_ids)
        ).delete(synchronize_session=False)

    signal_community_ids = set(
        db.scalars(select(SignalCommunity.id).where(SignalCommunity.signal_state_id.in_(signal_state_ids))).all()
    )
    if signal_community_ids:
        stats.removed_signal_community_memberships = db.query(SignalCommunityMembership).filter(
            SignalCommunityMembership.signal_community_id.in_(signal_community_ids)
        ).delete(synchronize_session=False)
        stats.removed_signal_communities = db.query(SignalCommunity).filter(
            SignalCommunity.id.in_(signal_community_ids)
        ).delete(synchronize_session=False)

    stats.removed_signal_edges = db.query(SignalEdge).filter(
        SignalEdge.signal_state_id.in_(signal_state_ids)
    ).delete(synchronize_session=False)
    stats.removed_signal_nodes = db.query(SignalNode).filter(
        SignalNode.signal_state_id.in_(signal_state_ids)
    ).delete(synchronize_session=False)
    stats.removed_signal_decisions = db.query(SignalDecision).filter(
        SignalDecision.signal_state_id.in_(signal_state_ids)
    ).delete(synchronize_session=False)
    stats.removed_signal_candidates = db.query(SignalCandidate).filter(
        SignalCandidate.signal_state_id.in_(signal_state_ids)
    ).delete(synchronize_session=False)
    stats.removed_signal_states = db.query(SignalState).filter(
        SignalState.id.in_(signal_state_ids)
    ).delete(synchronize_session=False)


def _delete_signal_schemas_for_knowledge_base(db: Session, stats: EvidenceCleanupStats, knowledge_base_id: str) -> None:
    schema_state_ids = set(
        db.scalars(select(SignalSchemaState.id).where(SignalSchemaState.knowledge_base_id == knowledge_base_id)).all()
    )
    if not schema_state_ids:
        return
    db.query(SignalState).filter(SignalState.schema_state_id.in_(schema_state_ids)).update(
        {"schema_state_id": None},
        synchronize_session=False,
    )
    db.query(SignalState).filter(
        SignalState.knowledge_base_id == knowledge_base_id,
        SignalState.evidence_community_state_id.is_not(None),
    ).update(
        {"evidence_community_state_id": None},
        synchronize_session=False,
    )
    db.query(SignalRelationSpec).filter(SignalRelationSpec.schema_state_id.in_(schema_state_ids)).delete(
        synchronize_session=False
    )
    db.query(SignalTypeSpec).filter(SignalTypeSpec.schema_state_id.in_(schema_state_ids)).delete(
        synchronize_session=False
    )
    stats.removed_signal_schema_states = db.query(SignalSchemaState).filter(
        SignalSchemaState.id.in_(schema_state_ids)
    ).delete(synchronize_session=False)


def cleanup_stale_graph(db: Session, knowledge_base_id: str) -> dict[str, int]:
    ensure_no_active_batch(db, knowledge_base_id)
    db.commit()
    return {
        "removed_evidence_edges": 0,
        "removed_evidence_atoms": 0,
        "removed_signal_nodes": 0,
        "removed_signal_edges": 0,
    }


def cleanup_stale_evidence_references(
    db: Session,
    *,
    knowledge_base_id: str,
    inactive_document_version_ids: set[str],
    inactive_chunk_ids: set[str],
    stale_qdrant_point_ids: set[str],
) -> EvidenceCleanupStats:
    stats = EvidenceCleanupStats()
    inactive_document_version_ids = {str(item) for item in inactive_document_version_ids if item}
    inactive_chunk_ids = {str(item) for item in inactive_chunk_ids if item}
    stale_qdrant_point_ids = {str(item) for item in stale_qdrant_point_ids if item}

    stale_atom_ids = set(
        db.scalars(
            select(EvidenceAtom.id).where(
                EvidenceAtom.knowledge_base_id == knowledge_base_id,
                or_(
                    EvidenceAtom.state != "active",
                    EvidenceAtom.document_version_id.in_(inactive_document_version_ids or {"__none__"}),
                ),
            )
        ).all()
    )
    stale_graph_state_ids = _graph_state_ids_touching_atoms(
        db,
        knowledge_base_id=knowledge_base_id,
        stale_atom_ids=stale_atom_ids,
        inactive_document_version_ids=inactive_document_version_ids,
    )

    candidate_ids = set()
    if stale_graph_state_ids:
        candidate_ids.update(
            db.scalars(select(ChunkCandidate.id).where(ChunkCandidate.graph_state_id.in_(stale_graph_state_ids))).all()
        )

    decision_ids = set()
    if stale_graph_state_ids:
        decision_ids.update(
            db.scalars(select(ChunkDecision.id).where(ChunkDecision.graph_state_id.in_(stale_graph_state_ids))).all()
        )
    if candidate_ids:
        decision_ids.update(
            db.scalars(select(ChunkDecision.id).where(ChunkDecision.candidate_id.in_(candidate_ids))).all()
        )

    stale_active_chunk_ids = set(inactive_chunk_ids)
    if decision_ids:
        stale_active_chunk_ids.update(
            db.scalars(select(ActiveChunk.id).where(ActiveChunk.chunk_decision_id.in_(decision_ids))).all()
        )
    stale_active_chunk_ids.update(
        db.scalars(
            select(ActiveChunk.id).where(
                ActiveChunk.knowledge_base_id == knowledge_base_id,
                ActiveChunk.state != "active",
            )
        ).all()
    )
    stale_active_chunk_ids.update(_active_chunk_ids_touching_atoms(db, knowledge_base_id, stale_atom_ids))

    if stale_active_chunk_ids or stale_qdrant_point_ids:
        conditions = []
        if stale_active_chunk_ids:
            conditions.append(VectorRecord.active_chunk_id.in_(stale_active_chunk_ids))
        if stale_qdrant_point_ids:
            conditions.append(VectorRecord.qdrant_point_id.in_(stale_qdrant_point_ids))
        stats.removed_vector_records = db.query(VectorRecord).filter(or_(*conditions)).delete(synchronize_session="fetch")

    if stale_graph_state_ids:
        signal_state_ids = set(
            db.scalars(select(SignalState.id).where(SignalState.evidence_graph_state_id.in_(stale_graph_state_ids))).all()
        )
        _delete_signal_and_projection_state(db, stats, signal_state_ids)

        community_state_ids = set(
            db.scalars(select(CommunityState.id).where(CommunityState.graph_state_id.in_(stale_graph_state_ids))).all()
        )
        if community_state_ids:
            stats.removed_community_summaries = db.query(CommunitySummary).filter(
                CommunitySummary.community_state_id.in_(community_state_ids)
            ).delete(synchronize_session=False)
            stats.removed_community_memberships = db.query(CommunityMembership).filter(
                CommunityMembership.community_state_id.in_(community_state_ids)
            ).delete(synchronize_session=False)
            stats.removed_community_states = db.query(CommunityState).filter(
                CommunityState.id.in_(community_state_ids)
            ).delete(synchronize_session=False)

    if stale_active_chunk_ids:
        stats.removed_active_chunks = db.query(ActiveChunk).filter(
            ActiveChunk.id.in_(stale_active_chunk_ids)
        ).delete(synchronize_session="fetch")

    if decision_ids:
        stats.removed_chunk_decisions = db.query(ChunkDecision).filter(
            ChunkDecision.id.in_(decision_ids)
        ).delete(synchronize_session="fetch")

    if candidate_ids:
        stats.removed_quality_decisions = db.query(QualityDecision).filter(
            QualityDecision.candidate_id.in_(candidate_ids)
        ).delete(synchronize_session="fetch")
        stats.removed_chunk_candidates = db.query(ChunkCandidate).filter(
            ChunkCandidate.id.in_(candidate_ids)
        ).delete(synchronize_session="fetch")

    edge_conditions = []
    if stale_graph_state_ids:
        edge_conditions.append(EvidenceEdge.graph_state_id.in_(stale_graph_state_ids))
    if stale_atom_ids:
        edge_conditions.extend(
            [
                EvidenceEdge.source_atom_id.in_(stale_atom_ids),
                EvidenceEdge.target_atom_id.in_(stale_atom_ids),
            ]
        )
    if edge_conditions:
        stats.removed_evidence_edges = db.query(EvidenceEdge).filter(or_(*edge_conditions)).delete(
            synchronize_session="fetch"
        )

    if stale_graph_state_ids:
        stats.removed_evidence_graph_states = db.query(EvidenceGraphState).filter(
            EvidenceGraphState.id.in_(stale_graph_state_ids)
        ).delete(synchronize_session="fetch")

    if stale_atom_ids:
        stats.removed_evidence_atoms = db.query(EvidenceAtom).filter(
            EvidenceAtom.id.in_(stale_atom_ids)
        ).delete(synchronize_session="fetch")

    return stats


def delete_document_graph_incremental(db: Session, knowledge_base_id: str, document_id: str) -> dict[str, int]:
    document_version_ids = set(
        db.scalars(select(DocumentVersion.id).where(DocumentVersion.document_id == document_id)).all()
    )
    active_chunk_ids = _active_chunk_ids_for_document(db, knowledge_base_id, document_id)
    stats = cleanup_stale_evidence_references(
        db,
        knowledge_base_id=knowledge_base_id,
        inactive_document_version_ids=document_version_ids,
        inactive_chunk_ids=active_chunk_ids,
        stale_qdrant_point_ids=set(),
    )
    deleted_parse_jobs = db.query(ParseJob).filter(ParseJob.document_id == document_id).delete(synchronize_session=False)
    deleted_source_files = db.query(SourceFile).filter(SourceFile.document_id == document_id).delete(synchronize_session=False)
    db.flush()
    return {
        "deleted_parse_jobs": deleted_parse_jobs,
        "deleted_source_files": deleted_source_files,
        **stats.as_dict(),
    }


def cleanup_stale_data(db: Session, knowledge_base_id: str, knowledge_base_name: str) -> dict[str, int]:
    from app.services.ingestion import create_vector_compensation_log, mark_vector_compensation_log

    ensure_no_active_batch(db, knowledge_base_id)

    active_chunk_ids = set(
        db.scalars(
            select(ActiveChunk.id).where(
                ActiveChunk.knowledge_base_id == knowledge_base_id,
                ActiveChunk.state == "active",
            )
        ).all()
    )
    vector_store = VectorStore(knowledge_base_name=knowledge_base_name)
    stale_vector_ids = sorted(set(vector_store.list_ids(knowledge_base_id)) - active_chunk_ids)

    inactive_document_ids = set(
        db.scalars(
            select(Document.id).where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.is_active.is_(False),
            )
        ).all()
    )
    if inactive_document_ids:
        db.query(IngestionJob).filter(IngestionJob.document_id.in_(inactive_document_ids)).update(
            {"document_id": None},
            synchronize_session="fetch",
        )

    inactive_version_ids = set(
        db.scalars(
            select(DocumentVersion.id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                or_(DocumentVersion.is_active.is_(False), Document.is_active.is_(False)),
            )
        ).all()
    )
    inactive_active_chunk_ids = set(
        db.scalars(
            select(ActiveChunk.id).where(
                ActiveChunk.knowledge_base_id == knowledge_base_id,
                ActiveChunk.state != "active",
            )
        ).all()
    )
    for document_id in inactive_document_ids:
        inactive_active_chunk_ids.update(_active_chunk_ids_for_document(db, knowledge_base_id, document_id))

    evidence_stats = cleanup_stale_evidence_references(
        db,
        knowledge_base_id=knowledge_base_id,
        inactive_document_version_ids=inactive_version_ids,
        inactive_chunk_ids=inactive_active_chunk_ids,
        stale_qdrant_point_ids=set(stale_vector_ids),
    )
    deleted_document_versions = (
        db.query(DocumentVersion)
        .filter(DocumentVersion.id.in_(inactive_version_ids))
        .delete(synchronize_session="fetch")
        if inactive_version_ids
        else 0
    )
    deleted_documents = db.query(Document).filter(
        Document.knowledge_base_id == knowledge_base_id,
        Document.is_active.is_(False),
    ).delete(synchronize_session="fetch")
    db.commit()

    if stale_vector_ids:
        delete_log = create_vector_compensation_log(
            db,
            knowledge_base_id=knowledge_base_id,
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
        "deleted_chunks": evidence_stats.removed_active_chunks,
        "deleted_document_versions": deleted_document_versions,
        "deleted_documents": deleted_documents,
        **evidence_stats.as_dict(),
    }


async def reconcile_vector_store(db: Session, knowledge_base_id: str | None = None, *, reembed_missing: bool = True) -> dict[str, int]:
    from app.services.chunking import CURRENT_EMBEDDING_TEXT_VERSION, contextual_embedding_text
    from app.services.embeddings import EmbeddingProvider
    from app.services.evidence_graph import stable_hash
    from app.services.ingestion import process_pending_vector_compensations

    processed_compensations = process_pending_vector_compensations(db)
    knowledge_bases = db.scalars(
        select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
        if knowledge_base_id
        else select(KnowledgeBase)
    ).all()
    stats = {
        "processed_compensations": processed_compensations,
        "scanned_knowledge_bases": len(knowledge_bases),
        "deleted_orphan_points": 0,
        "removed_orphan_vector_records": 0,
        "missing_points": 0,
        "reembedded_missing_points": 0,
        "upserted_vector_records": 0,
    }
    for knowledge_base in knowledge_bases:
        vector_store = VectorStore(knowledge_base_name=knowledge_base.name)
        qdrant_ids = set(vector_store.list_ids(knowledge_base.id))
        active_chunks = db.scalars(
            select(ActiveChunk).where(
                ActiveChunk.knowledge_base_id == knowledge_base.id,
                ActiveChunk.state == "active",
            )
        ).all()
        active_by_id = {chunk.id: chunk for chunk in active_chunks}
        active_ids = set(active_by_id)
        orphan_ids = sorted(qdrant_ids - active_ids)
        if orphan_ids:
            await vector_store.async_delete(orphan_ids)
            stats["deleted_orphan_points"] += len(orphan_ids)
            stats["removed_orphan_vector_records"] += db.query(VectorRecord).filter(
                VectorRecord.knowledge_base_id == knowledge_base.id,
                VectorRecord.qdrant_point_id.in_(orphan_ids),
            ).delete(synchronize_session=False)

        missing_ids = sorted(active_ids - qdrant_ids)
        if not missing_ids:
            db.commit()
            continue
        stats["missing_points"] += len(missing_ids)
        if not reembed_missing:
            db.commit()
            continue
        missing_chunks = [active_by_id[chunk_id] for chunk_id in missing_ids]
        embedding_inputs: list[str] = []
        payloads: list[dict] = []
        for chunk in missing_chunks:
            metadata = chunk.metadata_json or {}
            document = db.get(Document, metadata.get("document_id")) if metadata.get("document_id") else None
            document_title = document.title if document else str(metadata.get("document_title") or "Document")
            content_kind = metadata.get("content_kind")
            embedding_inputs.append(
                contextual_embedding_text(
                    document_title=document_title,
                    partition=metadata.get("partition"),
                    section=metadata.get("section"),
                    source_type=metadata.get("source_type"),
                    content_kind=content_kind,
                    content=chunk.text,
                    summary=metadata.get("summary"),
                    keywords=list(metadata.get("keywords") or []),
                    has_table=bool(metadata.get("has_table")),
                    has_formula=bool(metadata.get("has_formula")),
                )
            )
            payloads.append(
                {
                    "knowledge_base_id": knowledge_base.id,
                    "active_chunk_id": chunk.id,
                    "document_id": metadata.get("document_id"),
                    "document_title": document_title,
                    "source_path": document.source_path if document else metadata.get("source_path"),
                    "partition": metadata.get("partition"),
                    "section": metadata.get("section"),
                    "page_number": metadata.get("page_number"),
                    "snippet": metadata.get("snippet") or chunk.text[:240],
                    "source_type": metadata.get("source_type"),
                    "content": chunk.text,
                    "content_kind": content_kind,
                    "embedding_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
                    "evidence_atom_ids": list(chunk.atom_ids_json or []),
                    "source_span_union": dict(chunk.source_span_union_json or {}),
                    "graph_state_hash": chunk.graph_state_hash,
                    "quality_decision_id": chunk.quality_decision_id,
                    "policy_state_id": chunk.policy_state_id,
                    "community_ids": list(chunk.community_ids_json or []),
                }
            )
        embedding_result = await EmbeddingProvider().embed_texts_with_meta(embedding_inputs, text_type="document")
        points = [
            {"id": chunk.id, "vector": vector, "payload": payload}
            for chunk, vector, payload in zip(missing_chunks, embedding_result.vectors, payloads)
        ]
        await vector_store.async_upsert(points)
        stats["reembedded_missing_points"] += len(points)
        for chunk, point in zip(missing_chunks, points):
            payload_hash = stable_hash(point["payload"])
            db.query(VectorRecord).filter(VectorRecord.active_chunk_id == chunk.id).delete(synchronize_session=False)
            db.add(
                VectorRecord(
                    knowledge_base_id=knowledge_base.id,
                    active_chunk_id=chunk.id,
                    qdrant_point_id=chunk.id,
                    embedding_model=EmbeddingProvider().settings.embedding_model,
                    embedding_text_version=CURRENT_EMBEDDING_TEXT_VERSION,
                    payload_hash=payload_hash,
                    vector_status="ready",
                    diagnostics_json={"source": "vector_store_reconcile", "payload_hash": payload_hash},
                )
            )
            stats["upserted_vector_records"] += 1
        db.commit()
    return stats


def reconcile_vector_store_orphans(db: Session, knowledge_base_id: str | None = None) -> dict[str, int]:
    knowledge_bases = db.scalars(
        select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
        if knowledge_base_id
        else select(KnowledgeBase)
    ).all()
    stats = {
        "scanned_knowledge_bases": len(knowledge_bases),
        "deleted_orphan_points": 0,
        "removed_orphan_vector_records": 0,
        "missing_points": 0,
    }
    for knowledge_base in knowledge_bases:
        vector_store = VectorStore(knowledge_base_name=knowledge_base.name)
        qdrant_ids = set(vector_store.list_ids(knowledge_base.id))
        active_ids = set(
            db.scalars(
                select(ActiveChunk.id).where(
                    ActiveChunk.knowledge_base_id == knowledge_base.id,
                    ActiveChunk.state == "active",
                )
            ).all()
        )
        orphan_ids = sorted(qdrant_ids - active_ids)
        if orphan_ids:
            vector_store.delete(orphan_ids)
            stats["deleted_orphan_points"] += len(orphan_ids)
            stats["removed_orphan_vector_records"] += db.query(VectorRecord).filter(
                VectorRecord.knowledge_base_id == knowledge_base.id,
                VectorRecord.qdrant_point_id.in_(orphan_ids),
            ).delete(synchronize_session=False)
        stats["missing_points"] += len(active_ids - qdrant_ids)
        db.commit()
    return stats


def reconcile_vector_store_sync(db: Session, knowledge_base_id: str | None = None, *, reembed_missing: bool = True) -> dict[str, int]:
    return asyncio.run(reconcile_vector_store(db, knowledge_base_id=knowledge_base_id, reembed_missing=reembed_missing))


def reconcile_policy_state(db: Session, knowledge_base_id: str | None = None, *, cold_arm_event_threshold: int = 20) -> int:
    from app.services.evidence_graph import BANDIT_ARMS, BANDIT_CONTEXT_FEATURES, _normalize_policy_posterior

    query = select(PolicyState)
    if knowledge_base_id:
        query = query.where(PolicyState.knowledge_base_id == knowledge_base_id)
    states = db.scalars(query).all()
    reconciled = 0
    dimension = len(BANDIT_CONTEXT_FEATURES)
    for state in states:
        posterior = _normalize_policy_posterior(state)
        summary = dict(state.reward_summary_json or {})
        events = int(summary.get("events") or 0)
        changed = False
        if events >= cold_arm_event_threshold:
            for arm in BANDIT_ARMS:
                arm_state = posterior["arms"].setdefault(arm, {})
                if int(arm_state.get("count") or 0) == 0:
                    arm_state.update(
                        {
                            "A_diag": [1.0 for _ in range(dimension)],
                            "b": [0.0 for _ in range(dimension)],
                            "count": 0,
                            "reward_sum": 0.0,
                            "last_reward": None,
                            "token_cost_sum": 0.0,
                            "reconciled_reason": "cold_arm_prior_reset",
                        }
                    )
                    changed = True
        exploration = dict(state.exploration_json or {})
        alpha = float(exploration.get("alpha") or 0.25)
        if alpha > 0.5:
            exploration["alpha"] = 0.5
            exploration["warning"] = "alpha_capped_by_policy_reconcile"
            changed = True
        if changed:
            state.posterior_json = posterior
            state.exploration_json = exploration
            state.reward_summary_json = {
                **summary,
                "last_policy_reconcile_at": datetime.utcnow().isoformat(),
                "policy_reconcile_protocol": "policy_reconcile_v1",
            }
            reconciled += 1
    if reconciled:
        db.flush()
    return reconciled


def delete_knowledge_base_data(db: Session, knowledge_base: KnowledgeBase) -> dict[str, int]:
    ensure_no_active_batch(db, knowledge_base.id)
    settings = get_settings()
    knowledge_base_paths = settings.knowledge_base_paths_for_name(knowledge_base.name)
    data_root = settings.data_root.resolve()
    knowledge_base_root = knowledge_base_paths["knowledge_base_root"].resolve()
    if knowledge_base_root != data_root and data_root not in knowledge_base_root.parents:
        raise RuntimeError(f"Refusing to delete knowledge base directory outside DATA_ROOT: {knowledge_base_root}")

    vector_store = VectorStore(knowledge_base_name=knowledge_base.name)
    vector_ids = vector_store.list_ids(knowledge_base.id)

    run_ids = set(db.scalars(select(AgentRun.id).where(AgentRun.knowledge_base_id == knowledge_base.id)).all())
    batch_ids = set(db.scalars(select(IngestionBatch.id).where(IngestionBatch.knowledge_base_id == knowledge_base.id)).all())
    document_ids = set(db.scalars(select(Document.id).where(Document.knowledge_base_id == knowledge_base.id)).all())
    graph_state_ids = set(
        db.scalars(select(EvidenceGraphState.id).where(EvidenceGraphState.knowledge_base_id == knowledge_base.id)).all()
    )
    candidate_ids = (
        set(db.scalars(select(ChunkCandidate.id).where(ChunkCandidate.graph_state_id.in_(graph_state_ids))).all())
        if graph_state_ids
        else set()
    )
    signal_state_ids = set(db.scalars(select(SignalState.id).where(SignalState.knowledge_base_id == knowledge_base.id)).all())
    community_state_ids = set(db.scalars(select(CommunityState.id).where(CommunityState.knowledge_base_id == knowledge_base.id)).all())

    stats = EvidenceCleanupStats()
    deleted_reward_events = db.query(RewardEvent).filter(RewardEvent.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_policy_observations = db.query(PolicyObservation).filter(PolicyObservation.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_quality_observations = db.query(QualityObservation).filter(QualityObservation.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_citation_verifications = db.query(CitationVerification).filter(CitationVerification.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_answer_sessions = db.query(AnswerSession).filter(AnswerSession.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_retrieval_traces = db.query(RetrievalTrace).filter(RetrievalTrace.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_trace_events = db.query(AgentTraceEvent).filter(AgentTraceEvent.run_id.in_(run_ids)).delete(synchronize_session=False) if run_ids else 0
    deleted_agent_runs = db.query(AgentRun).filter(AgentRun.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_sessions = db.query(QASession).filter(QASession.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)

    deleted_vector_records = db.query(VectorRecord).filter(VectorRecord.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    _delete_signal_and_projection_state(db, stats, signal_state_ids)
    _delete_signal_schemas_for_knowledge_base(db, stats, knowledge_base.id)

    deleted_active_chunks = db.query(ActiveChunk).filter(ActiveChunk.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_chunk_decisions = db.query(ChunkDecision).filter(ChunkDecision.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_quality_decisions = (
        db.query(QualityDecision)
        .filter(QualityDecision.candidate_id.in_(candidate_ids or {"__none__"}))
        .delete(synchronize_session=False)
    )
    deleted_chunk_candidates = db.query(ChunkCandidate).filter(ChunkCandidate.id.in_(candidate_ids or {"__none__"})).delete(synchronize_session=False)

    deleted_community_summaries = db.query(CommunitySummary).filter(
        CommunitySummary.community_state_id.in_(community_state_ids)
    ).delete(synchronize_session=False) if community_state_ids else 0
    deleted_community_memberships = db.query(CommunityMembership).filter(
        CommunityMembership.community_state_id.in_(community_state_ids)
    ).delete(synchronize_session=False) if community_state_ids else 0
    deleted_community_states = db.query(CommunityState).filter(CommunityState.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_evidence_edges = db.query(EvidenceEdge).filter(
        EvidenceEdge.graph_state_id.in_(graph_state_ids or {"__none__"})
    ).delete(synchronize_session=False)
    deleted_evidence_graph_states = db.query(EvidenceGraphState).filter(EvidenceGraphState.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_evidence_atoms = db.query(EvidenceAtom).filter(EvidenceAtom.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_policy_states = db.query(PolicyState).filter(PolicyState.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)

    deleted_parse_jobs = db.query(ParseJob).filter(ParseJob.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_source_files = db.query(SourceFile).filter(SourceFile.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_ingestion_logs = db.query(IngestionLog).filter(IngestionLog.batch_id.in_(batch_ids)).delete(synchronize_session=False) if batch_ids else 0
    deleted_compensations = db.query(IngestionCompensationLog).filter(IngestionCompensationLog.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_jobs = db.query(IngestionJob).filter(IngestionJob.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)
    deleted_batches = db.query(IngestionBatch).filter(IngestionBatch.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)

    deleted_versions = db.query(DocumentVersion).filter(DocumentVersion.document_id.in_(document_ids)).delete(synchronize_session=False) if document_ids else 0
    deleted_documents = db.query(Document).filter(Document.knowledge_base_id == knowledge_base.id).delete(synchronize_session=False)

    db.delete(knowledge_base)
    db.commit()

    vector_store.delete(vector_ids)

    deleted_directory = 0
    if knowledge_base_root.exists():
        shutil.rmtree(knowledge_base_root)
        deleted_directory = 1

    return {
        "deleted_vectors": len(vector_ids),
        "deleted_vector_records": deleted_vector_records,
        "deleted_active_chunks": deleted_active_chunks,
        "deleted_chunk_decisions": deleted_chunk_decisions,
        "deleted_quality_decisions": deleted_quality_decisions,
        "deleted_chunk_candidates": deleted_chunk_candidates,
        "deleted_evidence_atoms": deleted_evidence_atoms,
        "deleted_evidence_edges": deleted_evidence_edges,
        "deleted_evidence_graph_states": deleted_evidence_graph_states,
        "deleted_community_states": deleted_community_states,
        "deleted_community_memberships": deleted_community_memberships,
        "deleted_community_summaries": deleted_community_summaries,
        "deleted_signal_schema_states": stats.removed_signal_schema_states,
        "deleted_signal_states": stats.removed_signal_states,
        "deleted_signal_candidates": stats.removed_signal_candidates,
        "deleted_signal_decisions": stats.removed_signal_decisions,
        "deleted_signal_nodes": stats.removed_signal_nodes,
        "deleted_signal_edges": stats.removed_signal_edges,
        "deleted_signal_communities": stats.removed_signal_communities,
        "deleted_signal_community_memberships": stats.removed_signal_community_memberships,
        "deleted_projection_states": stats.removed_projection_states,
        "deleted_projection_nodes": stats.removed_projection_nodes,
        "deleted_projection_edges": stats.removed_projection_edges,
        "deleted_projection_communities": stats.removed_projection_communities,
        "deleted_policy_states": deleted_policy_states,
        "deleted_policy_observations": deleted_policy_observations,
        "deleted_quality_observations": deleted_quality_observations,
        "deleted_retrieval_traces": deleted_retrieval_traces,
        "deleted_answer_sessions": deleted_answer_sessions,
        "deleted_citation_verifications": deleted_citation_verifications,
        "deleted_reward_events": deleted_reward_events,
        "deleted_trace_events": deleted_trace_events,
        "deleted_agent_runs": deleted_agent_runs,
        "deleted_sessions": deleted_sessions,
        "deleted_parse_jobs": deleted_parse_jobs,
        "deleted_source_files": deleted_source_files,
        "deleted_ingestion_logs": deleted_ingestion_logs,
        "deleted_compensations": deleted_compensations,
        "deleted_jobs": deleted_jobs,
        "deleted_batches": deleted_batches,
        "deleted_chunks": deleted_active_chunks,
        "deleted_document_versions": deleted_versions,
        "deleted_documents": deleted_documents,
        "deleted_knowledge_bases": 1,
        "deleted_directory": deleted_directory,
    }

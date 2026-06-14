from __future__ import annotations

import shutil
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import (
    BM25Record,
    Chunk,
    ChunkContextText,
    ChunkCoordinate,
    ChunkRelationEdge,
    ChunkSpan,
    ChunkStructureEdge,
    ChunkStructureMapping,
    ChunkStructureNode,
    ChunkVersion,
    CitationVerification,
    Document,
    DocumentVersion,
    FineClusterMembership,
    IngestionBatch,
    KnowledgeBase,
    PolicyState,
    VectorRecord,
)
from app.services.chunking import stable_hash
from app.services.vector_store import VectorStore


TERMINAL_BATCH_STATES = {"completed", "failed", "partial_failed", "skipped", "cancelled", "cancel_failed"}


class MaintenanceConflict(RuntimeError):
    pass


@dataclass
class VectorReconcileStats:
    knowledge_base_id: str | None
    checked_records: int = 0
    checked_collections: int = 0
    missing_points: int = 0
    stale_points: int = 0
    deleted_points: int = 0
    marked_records: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def active_batch_for_knowledge_base(db: Session, knowledge_base_id: str) -> IngestionBatch | None:
    return db.scalar(
        select(IngestionBatch)
        .where(IngestionBatch.knowledge_base_id == knowledge_base_id, IngestionBatch.status.notin_(TERMINAL_BATCH_STATES))
        .order_by(IngestionBatch.created_at.desc())
    )


def ensure_no_active_batch(db: Session, knowledge_base_id: str) -> None:
    if active_batch_for_knowledge_base(db, knowledge_base_id) is not None:
        raise MaintenanceConflict("Cannot run maintenance while an ingestion batch is active")


def delete_knowledge_base_data(db: Session, knowledge_base: KnowledgeBase) -> dict:
    ensure_no_active_batch(db, knowledge_base.id)
    vector_records = db.scalars(select(VectorRecord).where(VectorRecord.knowledge_base_id == knowledge_base.id)).all()
    points_by_collection: dict[str, list[str]] = defaultdict(list)
    for record in vector_records:
        points_by_collection[record.collection_name].append(record.qdrant_point_id)
    deleted_points = 0
    for collection_name, point_ids in points_by_collection.items():
        VectorStore(knowledge_base.name, collection_name=collection_name).delete(point_ids)
        deleted_points += len(point_ids)
    stats = {
        "documents": db.scalar(select(func.count(Document.id)).where(Document.knowledge_base_id == knowledge_base.id)) or 0,
        "chunks": db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id)) or 0,
        "vector_records": len(vector_records),
        "qdrant_points": deleted_points,
    }
    paths = get_settings().knowledge_base_paths_for_name(knowledge_base.name)
    db.delete(knowledge_base)
    db.commit()
    shutil.rmtree(paths["knowledge_base_root"], ignore_errors=True)
    return stats


def cleanup_stale_data(
    db: Session,
    knowledge_base_id: str,
    knowledge_base_name: str,
    *,
    dry_run: bool = True,
    delete_inactive_chunks: bool = False,
) -> dict:
    ensure_no_active_batch(db, knowledge_base_id)
    knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
    current_chunk_version = knowledge_base.current_chunk_version if knowledge_base else 0
    db.execute(
        update(ChunkVersion)
        .where(ChunkVersion.knowledge_base_id == knowledge_base_id, ChunkVersion.chunk_version != current_chunk_version, ChunkVersion.state == "active")
        .values(state="inactive")
    )
    db.execute(
        update(ChunkVersion)
        .where(ChunkVersion.knowledge_base_id == knowledge_base_id, ChunkVersion.chunk_version == current_chunk_version, ChunkVersion.state != "active")
        .values(state="active")
    )
    inactive_chunks = db.scalars(select(Chunk.id).where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.state != "active")).all()
    inactive_chunk_ids = set(inactive_chunks)
    inactive_document_version_ids = {
        row[0]
        for row in db.execute(
            select(DocumentVersion.id)
            .join(Document, DocumentVersion.document_id == Document.id)
            .where(Document.knowledge_base_id == knowledge_base_id, DocumentVersion.is_active.is_(False))
        ).all()
    }
    inactive_chunk_version_ids = {
        row[0]
        for row in db.execute(
            select(ChunkVersion.id).where(
                ChunkVersion.knowledge_base_id == knowledge_base_id,
                ChunkVersion.chunk_version != current_chunk_version,
            )
        ).all()
    }
    inactive_chunk_id_filter = inactive_chunk_ids or {"__none__"}
    inactive_document_version_filter = inactive_document_version_ids or {"__none__"}
    stale_records = db.scalars(select(VectorRecord).where(VectorRecord.knowledge_base_id == knowledge_base_id, VectorRecord.chunk_id.in_(inactive_chunk_ids or {"__none__"}))).all()
    stale_bm25_records = db.scalars(select(BM25Record).where(BM25Record.knowledge_base_id == knowledge_base_id, BM25Record.chunk_id.in_(inactive_chunk_id_filter))).all()
    points_by_collection: dict[str, list[str]] = defaultdict(list)
    for record in stale_records:
        points_by_collection[record.collection_name].append(record.qdrant_point_id)
    inactive_counts = {
        "chunk_spans": db.scalar(select(func.count(ChunkSpan.id)).where(ChunkSpan.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_coordinates": db.scalar(select(func.count(ChunkCoordinate.id)).where(ChunkCoordinate.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_context_texts": db.scalar(select(func.count(ChunkContextText.id)).where(ChunkContextText.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_structure_mappings": db.scalar(select(func.count(ChunkStructureMapping.id)).where(ChunkStructureMapping.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_relation_edges": db.scalar(
            select(func.count(ChunkRelationEdge.id)).where(
                (ChunkRelationEdge.source_chunk_id.in_(inactive_chunk_id_filter)) | (ChunkRelationEdge.target_chunk_id.in_(inactive_chunk_id_filter))
            )
        )
        or 0,
        "fine_cluster_memberships": db.scalar(select(func.count(FineClusterMembership.id)).where(FineClusterMembership.chunk_id.in_(inactive_chunk_id_filter))) or 0,
        "chunk_structure_edges": db.scalar(select(func.count(ChunkStructureEdge.id)).where(ChunkStructureEdge.document_version_id.in_(inactive_document_version_filter))) or 0,
        "chunk_structure_nodes": db.scalar(select(func.count(ChunkStructureNode.id)).where(ChunkStructureNode.document_version_id.in_(inactive_document_version_filter))) or 0,
    }
    deleted_inactive = {
        "chunks": 0,
        "document_versions": 0,
        "chunk_versions": 0,
    }
    if not dry_run:
        for collection_name, point_ids in points_by_collection.items():
            VectorStore(knowledge_base_name, collection_name=collection_name).delete(point_ids)
        if stale_records:
            db.query(VectorRecord).filter(VectorRecord.id.in_([record.id for record in stale_records])).delete(synchronize_session=False)
        if stale_bm25_records:
            db.query(BM25Record).filter(BM25Record.id.in_([record.id for record in stale_bm25_records])).delete(synchronize_session=False)
        if delete_inactive_chunks and inactive_chunk_ids:
            db.execute(update(CitationVerification).where(CitationVerification.chunk_id.in_(inactive_chunk_id_filter)).values(chunk_id=None))
            db.query(ChunkRelationEdge).filter(
                (ChunkRelationEdge.source_chunk_id.in_(inactive_chunk_id_filter)) | (ChunkRelationEdge.target_chunk_id.in_(inactive_chunk_id_filter))
            ).delete(synchronize_session=False)
            db.query(FineClusterMembership).filter(FineClusterMembership.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkStructureMapping).filter(ChunkStructureMapping.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkContextText).filter(ChunkContextText.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkCoordinate).filter(ChunkCoordinate.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            db.query(ChunkSpan).filter(ChunkSpan.chunk_id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
            deleted_inactive["chunks"] = db.query(Chunk).filter(Chunk.id.in_(inactive_chunk_id_filter)).delete(synchronize_session=False)
        if delete_inactive_chunks and inactive_document_version_ids:
            db.query(ChunkStructureEdge).filter(ChunkStructureEdge.document_version_id.in_(inactive_document_version_filter)).delete(synchronize_session=False)
            db.query(ChunkStructureNode).filter(ChunkStructureNode.document_version_id.in_(inactive_document_version_filter)).delete(synchronize_session=False)
            deleted_inactive["document_versions"] = db.query(DocumentVersion).filter(DocumentVersion.id.in_(inactive_document_version_filter)).delete(synchronize_session=False)
        if delete_inactive_chunks and inactive_chunk_version_ids:
            deleted_inactive["chunk_versions"] = db.query(ChunkVersion).filter(ChunkVersion.id.in_(inactive_chunk_version_ids)).delete(synchronize_session=False)
        db.commit()
    else:
        db.rollback()
    return {
        "inactive_chunks": len(inactive_chunk_ids),
        "inactive_document_versions": len(inactive_document_version_ids),
        "inactive_chunk_versions": len(inactive_chunk_version_ids),
        "stale_vector_records": len(stale_records),
        "stale_bm25_records": len(stale_bm25_records),
        "stale_qdrant_points": sum(len(ids) for ids in points_by_collection.values()),
        "collections": sorted(points_by_collection),
        "delete_inactive_chunks": delete_inactive_chunks,
        "inactive_dependents": inactive_counts,
        "deleted_inactive": deleted_inactive,
        "applied": not dry_run,
    }


def reconcile_vector_store_sync(db: Session, knowledge_base_id: str | None = None) -> dict:
    stats = VectorReconcileStats(knowledge_base_id=knowledge_base_id)
    query = select(VectorRecord)
    if knowledge_base_id:
        query = query.where(VectorRecord.knowledge_base_id == knowledge_base_id)
    records = db.scalars(query).all()
    stats.checked_records = len(records)
    records_by_collection: dict[str, list[VectorRecord]] = defaultdict(list)
    for record in records:
        records_by_collection[record.collection_name].append(record)
    for collection_name, collection_records in records_by_collection.items():
        stats.checked_collections += 1
        kb_name = None
        if collection_records:
            kb = db.get(KnowledgeBase, collection_records[0].knowledge_base_id)
            kb_name = kb.name if kb else None
        store = VectorStore(kb_name, collection_name=collection_name)
        active_chunk_ids = {
            chunk_id
            for (chunk_id,) in db.execute(
                select(Chunk.id).where(
                    Chunk.id.in_([record.chunk_id for record in collection_records] or {"__none__"}),
                    Chunk.state == "active",
                )
            ).all()
        }
        health = store.health_check(collection_records[0].knowledge_base_id if collection_records else knowledge_base_id or "", list(active_chunk_ids))
        missing = set(health.get("missing") or [])
        stale = set(health.get("stale") or [])
        stats.missing_points += len(missing)
        stats.stale_points += len(stale)
        for record in collection_records:
            desired = "ready"
            if record.qdrant_point_id in missing:
                desired = "missing"
            elif record.qdrant_point_id in stale or record.chunk_id not in active_chunk_ids:
                desired = "stale"
            if record.vector_status != desired:
                record.vector_status = desired
                stats.marked_records += 1
    db.commit()
    return stats.as_dict()


def reconcile_policy_state(db: Session, knowledge_base_id: str) -> int:
    latest = db.scalar(
        select(PolicyState)
        .where(PolicyState.knowledge_base_id == knowledge_base_id, PolicyState.policy_family == "context_graph_bandit")
        .order_by(PolicyState.created_at.desc())
    )
    arms = [
        "high_precision_direct_chunk",
        "structure_context_heavy",
        "fine_cluster_expansion",
        "mid_concept_expansion",
        "coarse_to_mid_drilldown",
        "bridge_edge_exploration",
        "formula_table_closure",
        "cross_document_synthesis",
        "low_latency_minimal_context",
    ]
    weights = {arm: 1.0 for arm in arms}
    state_hash = stable_hash({"policy_version": "context_graph_bandit_v1", "weights": weights})
    if latest and latest.state_hash == state_hash:
        return 0
    db.add(
        PolicyState(
            knowledge_base_id=knowledge_base_id,
            policy_family="context_graph_bandit",
            policy_version="context_graph_bandit_v1",
            weights_json=weights,
            constraints_json={"fallback_disabled": True, "citation_verification_required": True},
            exploration_json={"epsilon": 0.05, "max_bridge_expansion": 4},
            reward_summary_json={},
            state_hash=state_hash,
        )
    )
    return 1

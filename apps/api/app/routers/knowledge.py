from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ActiveChunk, AgentRun, AgentTraceEvent, ChunkCandidate, ChunkDecision, Document, EvidenceAtom, EvidenceEdge, EvidenceGraphState, IngestionBatch, IngestionJob, PolicyState, QASession, QualityDecision, SignalNode
from app.core.config import get_settings
from app.schemas import (
    AgentRequest,
    AgentResponse,
    BatchLogTokenResponse,
    BatchStartResponse,
    CleanupStaleDataResponse,
    KnowledgeBaseFileSummary,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseSummary,
    DashboardSnapshot,
    DeleteKnowledgeBaseResponse,
    DeleteResponse,
    ChunkDecisionOut,
    EvidenceAtomOut,
    RebuildGraphResponse,
    GraphResponse,
    IngestionBatchSummary,
    JobStatusResponse,
    ModelSettingsResponse,
    ModelSettingsUpdate,
    ParseUploadedFilesRequest,
    QARequest,
    QAResponse,
    QueryEvidenceGraphRequest,
    PolicyStateOut,
    QualityDecisionOut,
    RebuildGraphRequest,
    RefreshResponse,
    RuntimeCheckResponse,
    SearchRequest,
    SearchResponse,
    SessionMessagesResponse,
    SessionSummary,
    StrategyProfileBindRequest,
    StrategyProfileAssistantRequest,
    StrategyProfileAssistantStateResponse,
    StrategyProfileCopyRequest,
    StrategyProfileCreateRequest,
    StrategyProfileDetail,
    StrategyProfileDraftRequest,
    StrategyProfileDraftResponse,
    StrategyProfileMutationResponse,
    StrategyProfileSummary,
    StrategyProfileUpdateRequest,
    SignalNodeOut,
    TaskStatusResponse,
    UploadFileResponse,
)
from app.services.evidence_graph_payload import get_graph_payload, get_query_evidence_graph_payload
from app.services.evidence_graph import publish_global_evidence_graph_state, update_policy_from_rewards
from app.services.embeddings import is_degraded_mode
from app.services.agent_graph import run_agent, run_to_task_status, stream_agent_events
from app.services.ingestion import (
    create_knowledge_base_space,
    collect_source_documents,
    create_uploaded_files_batch,
    create_job,
    create_sync_batch,
    get_batch_status,
    list_knowledge_base_summaries,
    request_batch_cancel_control,
    register_uploaded_file,
    resolve_knowledge_base,
    run_batch_ingestion,
    run_ingestion_job,
    run_uploaded_files_ingestion,
    remove_knowledge_base_file,
    summarize_knowledge_base,
)
from app.services.ingestion_logs import (
    TERMINAL_LOG_EVENTS,
    create_log_stream_token,
    list_ingestion_logs,
    subscribe_ingestion_logs,
    unsubscribe_ingestion_logs,
    validate_log_stream_token,
)
from app.services.maintenance import MaintenanceConflict, cleanup_stale_data, delete_knowledge_base_data, reconcile_policy_state
from app.services.retrieval import (
    evidence_first_search_chunks_with_audit,
    get_dashboard_snapshot,
    get_job_status,
    list_knowledge_base_files,
    search_chunks_with_audit,
)
from app.services.runtime_settings import model_settings_payload, runtime_check_payload, update_model_settings
from app.services.storage import save_upload
from app.services.strategy_profiles import (
    bind_profile_to_knowledge_base,
    copy_profile,
    create_profile,
    delete_profile,
    generate_profile_draft,
    get_profile_or_raise,
    list_profiles,
    profile_to_payload,
    update_profile,
)
from app.services.profile_assistant import get_profile_assistant_state, stream_profile_assistant_events

router = APIRouter()

def get_requested_knowledge_base(db: Session, knowledge_base_id: str | None = None):
    try:
        return resolve_knowledge_base(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc



@router.get("/knowledge_bases", response_model=list[KnowledgeBaseSummary])
def list_knowledge_bases(db: Session = Depends(get_db)) -> list[dict]:
    return list_knowledge_base_summaries(db)


@router.post("/knowledge_bases", response_model=KnowledgeBaseSummary)
def create_knowledge_base(request: KnowledgeBaseCreateRequest, db: Session = Depends(get_db)) -> dict:
    try:
        KnowledgeBase = create_knowledge_base_space(db, request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summarize_knowledge_base(db, KnowledgeBase)


@router.delete("/knowledge_bases/{knowledge_base_id}", response_model=DeleteKnowledgeBaseResponse)
def delete_knowledge_base(knowledge_base_id: str, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        stats = delete_knowledge_base_data(db, KnowledgeBase)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True, **stats}


@router.get("/knowledge_bases/current/dashboard", response_model=DashboardSnapshot)
@router.get("/knowledge_bases/default/dashboard", response_model=DashboardSnapshot, include_in_schema=False)
def knowledge_base_dashboard(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    return get_dashboard_snapshot(db, KnowledgeBase.id)


@router.post("/knowledge_bases/current/refresh", response_model=RefreshResponse)
def refresh_current_knowledge_base(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    return {"knowledge_base_id": KnowledgeBase.id, "refreshed_at": datetime.utcnow()}


@router.get("/knowledge_bases/{knowledge_base_id}/evidence-atoms", response_model=list[EvidenceAtomOut])
def list_evidence_atoms(knowledge_base_id: str, offset: int = 0, limit: int = 100, db: Session = Depends(get_db)) -> list[dict]:
    get_requested_knowledge_base(db, knowledge_base_id)
    atoms = db.scalars(
        select(EvidenceAtom)
        .where(EvidenceAtom.knowledge_base_id == knowledge_base_id)
        .order_by(EvidenceAtom.created_at.desc(), EvidenceAtom.id.asc())
        .offset(max(offset, 0))
        .limit(min(max(limit, 1), 500))
    ).all()
    return [
        {
            "id": atom.id,
            "knowledge_base_id": atom.knowledge_base_id,
            "document_id": atom.document_id,
            "document_version_id": atom.document_version_id,
            "atom_type": atom.atom_type,
            "text": atom.text,
            "source_span_json": atom.source_span_json or {},
            "state": atom.state,
            "created_at": atom.created_at,
        }
        for atom in atoms
    ]


@router.get("/knowledge_bases/{knowledge_base_id}/signal-nodes", response_model=list[SignalNodeOut])
def list_signal_nodes(knowledge_base_id: str, offset: int = 0, limit: int = 100, db: Session = Depends(get_db)) -> list[dict]:
    get_requested_knowledge_base(db, knowledge_base_id)
    nodes = db.scalars(
        select(SignalNode)
        .where(SignalNode.knowledge_base_id == knowledge_base_id)
        .order_by(SignalNode.created_at.desc(), SignalNode.id.asc())
        .offset(max(offset, 0))
        .limit(min(max(limit, 1), 500))
    ).all()
    return [
        {
            "id": node.id,
            "knowledge_base_id": node.knowledge_base_id,
            "signal_state_id": node.signal_state_id,
            "canonical_label": node.canonical_label,
            "signal_type": node.signal_type,
            "support_atom_ids": list(node.support_atom_ids_json or []),
            "support_active_chunk_ids": list(node.support_active_chunk_ids_json or []),
            "source_span_union": node.source_span_union_json or {},
            "confidence": node.confidence,
            "created_at": node.created_at,
        }
        for node in nodes
    ]


@router.get("/knowledge_bases/{knowledge_base_id}/quality-decisions", response_model=list[QualityDecisionOut])
def list_quality_decisions(knowledge_base_id: str, active_chunk_id: str | None = None, offset: int = 0, limit: int = 100, db: Session = Depends(get_db)) -> list[QualityDecision]:
    get_requested_knowledge_base(db, knowledge_base_id)
    query = (
        select(QualityDecision)
        .join(ChunkCandidate, ChunkCandidate.id == QualityDecision.candidate_id)
        .join(EvidenceGraphState, EvidenceGraphState.id == ChunkCandidate.graph_state_id)
        .where(EvidenceGraphState.knowledge_base_id == knowledge_base_id)
    )
    if active_chunk_id:
        active_chunk = db.get(ActiveChunk, active_chunk_id)
        if active_chunk is None or active_chunk.knowledge_base_id != knowledge_base_id:
            raise HTTPException(status_code=404, detail="Active chunk not found")
        query = query.where(QualityDecision.id == active_chunk.quality_decision_id)
    return db.scalars(
        query.order_by(QualityDecision.created_at.desc(), QualityDecision.id.asc())
        .offset(max(offset, 0))
        .limit(min(max(limit, 1), 500))
    ).all()


@router.get("/knowledge_bases/{knowledge_base_id}/chunk-decisions", response_model=list[ChunkDecisionOut])
def list_chunk_decisions(knowledge_base_id: str, offset: int = 0, limit: int = 100, db: Session = Depends(get_db)) -> list[ChunkDecision]:
    get_requested_knowledge_base(db, knowledge_base_id)
    return db.scalars(
        select(ChunkDecision)
        .where(ChunkDecision.knowledge_base_id == knowledge_base_id)
        .order_by(ChunkDecision.created_at.desc(), ChunkDecision.id.asc())
        .offset(max(offset, 0))
        .limit(min(max(limit, 1), 500))
    ).all()


@router.get("/knowledge_bases/{knowledge_base_id}/policy-states", response_model=list[PolicyStateOut])
def list_policy_states(knowledge_base_id: str, offset: int = 0, limit: int = 50, db: Session = Depends(get_db)) -> list[PolicyState]:
    get_requested_knowledge_base(db, knowledge_base_id)
    return db.scalars(
        select(PolicyState)
        .where(PolicyState.knowledge_base_id == knowledge_base_id)
        .order_by(PolicyState.created_at.desc(), PolicyState.id.asc())
        .offset(max(offset, 0))
        .limit(min(max(limit, 1), 200))
    ).all()


@router.get("/knowledge-base-files", response_model=list[KnowledgeBaseFileSummary])
def knowledge_base_files(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    return list_knowledge_base_files(db, KnowledgeBase.id)


@router.delete("/knowledge-base-files")
def delete_knowledge_base_file(source_path: str, knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    if not remove_knowledge_base_file(db, KnowledgeBase, source_path):
        raise HTTPException(status_code=404, detail="File not found")
    return {"removed": True}


@router.post("/maintenance/cleanup-stale-data", response_model=CleanupStaleDataResponse)
def cleanup_knowledge_base_stale_data(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        return cleanup_stale_data(db, KnowledgeBase.id, KnowledgeBase.name)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/maintenance/policy-reconcile")
def reconcile_policy_endpoint(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    knowledge_base_ids = [knowledge_base_id] if knowledge_base_id else list(db.scalars(select(KnowledgeBase.id)).all())
    updated = 0
    reconciled = 0
    for kb_id in knowledge_base_ids:
        updated += update_policy_from_rewards(db, knowledge_base_id=kb_id)
        reconciled += reconcile_policy_state(db, knowledge_base_id=kb_id)
    db.commit()
    return {
        "knowledge_bases": len(knowledge_base_ids),
        "updated_observations": updated,
        "reconciled_policy_states": reconciled,
    }


@router.post("/maintenance/rebuild-graph", response_model=RebuildGraphResponse)
async def rebuild_graph_endpoint(
    request: RebuildGraphRequest = RebuildGraphRequest(),
    knowledge_base_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    affected_documents = db.scalar(select(func.count(Document.id)).where(Document.knowledge_base_id == KnowledgeBase.id)) or 0
    graph_stats: dict = {}
    if not request.dry_run:
        result = publish_global_evidence_graph_state(db, knowledge_base_id=KnowledgeBase.id)
        graph_stats = result.stats
        db.commit()
    latest_graph_state = db.scalar(
        select(EvidenceGraphState)
        .where(
            EvidenceGraphState.knowledge_base_id == KnowledgeBase.id,
            EvidenceGraphState.scope_type == "global",
            EvidenceGraphState.state == "active",
        )
        .order_by(EvidenceGraphState.created_at.desc())
    )
    evidence_edges = (
        db.scalar(select(func.count(EvidenceEdge.id)).where(EvidenceEdge.graph_state_id == latest_graph_state.id)) or 0
        if latest_graph_state is not None
        else 0
    )
    return {
        "batch_id": None,
        "state": "evidence_graph_active",
        "mode": "evidence",
        "affected_documents": affected_documents,
        "dry_run": request.dry_run,
        "evidence_atoms": db.scalar(select(func.count(EvidenceAtom.id)).where(EvidenceAtom.knowledge_base_id == KnowledgeBase.id, EvidenceAtom.state == "active")) or 0,
        "evidence_edges": evidence_edges,
        "active_chunks": db.scalar(select(func.count(ActiveChunk.id)).where(ActiveChunk.knowledge_base_id == KnowledgeBase.id, ActiveChunk.state == "active")) or 0,
        **graph_stats,
    }


@router.get("/knowledge_bases/current/graph", response_model=GraphResponse)
@router.get("/knowledge_bases/default/graph", response_model=GraphResponse, include_in_schema=False)
def knowledge_base_graph(knowledge_base_id: str | None = None, graph_type: str | None = None, view: str | None = None, db: Session = Depends(get_db)) -> dict:
    if graph_type != "evidence":
        raise HTTPException(status_code=400, detail="graph_type is required and must be: evidence")
    if view is not None and view not in {"overview", "detail", "neighborhood"}:
        raise HTTPException(status_code=400, detail="view must be one of: overview, detail, neighborhood")
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    return get_graph_payload(db, KnowledgeBase.id, graph_type=graph_type, view=view)


@router.get("/graph/partitions/{partition}", response_model=GraphResponse)
def partition_graph(partition: str, knowledge_base_id: str | None = None, graph_type: str | None = None, view: str | None = None, db: Session = Depends(get_db)) -> dict:
    if graph_type != "evidence":
        raise HTTPException(status_code=400, detail="graph_type is required and must be: evidence")
    if view is not None and view not in {"overview", "detail", "neighborhood"}:
        raise HTTPException(status_code=400, detail="view must be one of: overview, detail, neighborhood")
    KnowledgeBase = get_requested_knowledge_base(db, knowledge_base_id)
    return get_graph_payload(db, KnowledgeBase.id, partition=partition, graph_type=graph_type, view=view)


@router.post("/search/evidence-graph", response_model=GraphResponse)
def search_evidence_graph(request: QueryEvidenceGraphRequest, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, request.knowledge_base_id)
    return get_query_evidence_graph_payload(db, KnowledgeBase.id, request.chunk_ids, query=request.query)

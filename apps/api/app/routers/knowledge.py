from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Chunk, Document
from app.schemas import (
    CleanupStaleDataResponse,
    ContextPackageResponse,
    DashboardSnapshot,
    DeleteKnowledgeBaseResponse,
    GraphResponse,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseFileSummary,
    KnowledgeBaseSummary,
    RebuildGraphRequest,
    RebuildGraphResponse,
    RefreshResponse,
    RetrievalTraceStepsResponse,
)
from app.services.context_graph import context_graph_stats, graph_layer_payload, rebuild_context_graph
from app.services.ingestion import (
    create_knowledge_base_space,
    list_knowledge_base_files,
    list_knowledge_base_summaries,
    remove_knowledge_base_file,
    resolve_knowledge_base,
    summarize_knowledge_base,
)
from app.services.maintenance import MaintenanceConflict, cleanup_stale_data, delete_knowledge_base_data
from app.services.retrieval import get_context_package, get_dashboard_snapshot, get_retrieval_trace_steps

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
        knowledge_base = create_knowledge_base_space(db, request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summarize_knowledge_base(db, knowledge_base)


@router.delete("/knowledge_bases/{knowledge_base_id}", response_model=DeleteKnowledgeBaseResponse)
def delete_knowledge_base(knowledge_base_id: str, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        stats = delete_knowledge_base_data(db, knowledge_base)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True, "knowledge_base_id": knowledge_base_id, "knowledge_base_name": knowledge_base.name, "stats": stats}


@router.get("/knowledge_bases/current/dashboard", response_model=DashboardSnapshot)
@router.get("/knowledge_bases/default/dashboard", response_model=DashboardSnapshot, include_in_schema=False)
def knowledge_base_dashboard(knowledge_base_id: str | None = None, include_graph: bool = True, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    return get_dashboard_snapshot(db, knowledge_base.id, include_graph=include_graph)


@router.post("/knowledge_bases/current/refresh", response_model=RefreshResponse)
def refresh_current_knowledge_base(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    return {"knowledge_base_id": knowledge_base.id, "refreshed_at": datetime.utcnow()}


@router.get("/knowledge-base-files", response_model=list[KnowledgeBaseFileSummary])
def knowledge_base_files(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    return list_knowledge_base_files(db, knowledge_base.id)


@router.delete("/knowledge-base-files")
def delete_knowledge_base_file(source_path: str, knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    if not remove_knowledge_base_file(db, knowledge_base, source_path):
        raise HTTPException(status_code=404, detail="File not found")
    return {"removed": True}


@router.get("/knowledge_bases/{knowledge_base_id}/context-graph/stats")
def get_context_graph_stats(knowledge_base_id: str, db: Session = Depends(get_db)) -> dict:
    get_requested_knowledge_base(db, knowledge_base_id)
    return context_graph_stats(db, knowledge_base_id)


@router.get("/knowledge_bases/{knowledge_base_id}/graph/{graph_type}", response_model=GraphResponse)
def knowledge_base_layer_graph(knowledge_base_id: str, graph_type: str, limit: int = 200, db: Session = Depends(get_db)) -> dict:
    get_requested_knowledge_base(db, knowledge_base_id)
    try:
        return graph_layer_payload(db, knowledge_base_id, graph_type, limit=min(max(limit, 1), 1000))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/knowledge_bases/current/graph", response_model=GraphResponse)
@router.get("/knowledge_bases/default/graph", response_model=GraphResponse, include_in_schema=False)
def current_knowledge_base_graph(knowledge_base_id: str | None = None, graph_type: str = "chunk-relation", limit: int = 200, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        return graph_layer_payload(db, knowledge_base.id, graph_type, limit=min(max(limit, 1), 1000))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/context-packages/{package_id}", response_model=ContextPackageResponse)
def context_package(package_id: str, db: Session = Depends(get_db)) -> dict:
    payload = get_context_package(db, package_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Context package not found")
    return payload


@router.get("/retrieval-traces/{trace_id}/graph-steps", response_model=RetrievalTraceStepsResponse)
def retrieval_trace_steps(trace_id: str, db: Session = Depends(get_db)) -> dict:
    payload = get_retrieval_trace_steps(db, trace_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Retrieval trace not found")
    return payload


@router.post("/maintenance/cleanup-stale-data", response_model=CleanupStaleDataResponse)
def cleanup_knowledge_base_stale_data(
    knowledge_base_id: str | None = None,
    dry_run: bool = True,
    delete_inactive_chunks: bool = False,
    db: Session = Depends(get_db),
) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        stats = cleanup_stale_data(db, knowledge_base.id, knowledge_base.name, dry_run=dry_run, delete_inactive_chunks=delete_inactive_chunks)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"knowledge_base_id": knowledge_base.id, "dry_run": dry_run, "delete_inactive_chunks": delete_inactive_chunks, "stats": stats}


@router.post("/maintenance/rebuild-graph", response_model=RebuildGraphResponse)
@router.post("/maintenance/rebuild-context-graph", response_model=RebuildGraphResponse)
async def rebuild_graph_endpoint(
    request: RebuildGraphRequest = RebuildGraphRequest(),
    knowledge_base_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    affected_documents = db.scalar(select(func.count(Document.id)).where(Document.knowledge_base_id == knowledge_base.id, Document.is_active.is_(True))) or 0
    active_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
    if request.dry_run:
        return {
            "batch_id": None,
            "state": "dry_run",
            "mode": "four_layer_context_graph",
            "affected_documents": affected_documents,
            "dry_run": True,
            "stats": {"active_chunks": active_chunks, "layers": request.layers},
        }
    context_state = await rebuild_context_graph(db, knowledge_base.id)
    db.commit()
    return {
        "batch_id": context_state.build_batch_id,
        "state": context_state.status,
        "mode": "four_layer_context_graph",
        "affected_documents": affected_documents,
        "dry_run": False,
        "stats": context_state.stats_json or {},
    }

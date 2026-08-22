from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import get_db
from app.models import Chunk, Document, IngestionBatch
from app.schemas import (
    CleanupStaleDataResponse,
    ContextPackageResponse,
    DashboardSnapshot,
    DeleteKnowledgeBaseResponse,
    GraphFreshness,
    GraphResponse,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseFileSummary,
    KnowledgeBaseSummary,
    RebuildGraphRequest,
    RebuildGraphResponse,
    RefreshResponse,
    RetrievalTraceStepsResponse,
    AutoTpeStatusResponse,
)
from app.services.context_graph import (
    EntrySelectionTraceInvariantError,
    context_graph_stats,
    graph_layer_payload,
    reconcile_context_graph_freshness,
)
from app.services.ingestion import (
    create_context_graph_rebuild_batch,
    create_knowledge_base_space,
    exception_message,
    list_knowledge_base_files,
    list_knowledge_base_summaries,
    remove_knowledge_base_file,
    resolve_knowledge_base,
    run_context_graph_rebuild_batch,
    SOURCE_FILE_DELETE_LOCK_OPERATION,
    SourceFileDeleteError,
    source_file_delete_lock_token,
    summarize_knowledge_base,
)
from app.services.ingestion_logs import emit_ingestion_log
from app.services.ingestion_resource_lock import (
    IngestionResourceBusyError,
    knowledge_base_delete_recovery_owner_token,
    knowledge_base_ingestion_resource_lock,
)
from app.services.maintenance import (
    KNOWLEDGE_BASE_DELETE_LOCK_OPERATION,
    MaintenanceConflict,
    cleanup_stale_data,
    delete_knowledge_base_data,
)
from app.services.retrieval import (
    ContextPackagePublicIntegrityError,
    RetrievalTraceAuditError,
    get_context_package,
    get_dashboard_snapshot,
    get_retrieval_trace_steps,
)
from app.services.auto_tpe import latest_auto_tpe_status
from app.services.storage import UploadValidationError, run_bounded_source_io

router = APIRouter()


def get_requested_knowledge_base(db: Session, knowledge_base_id: str | None = None):
    try:
        return resolve_knowledge_base(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def mark_graph_rebuild_enqueued(db: Session, batch: IngestionBatch, task_id: str | None) -> None:
    stats = dict(batch.stats or {})
    stats["ingestion_execution_mode"] = "celery"
    stats["celery_task_name"] = "rebuild_context_graph_batch"
    if task_id:
        stats["celery_task_id"] = task_id
        stats["batch_task_ids"] = sorted({*stats.get("batch_task_ids", []), str(task_id)})
        stats["task_last_seen_at"] = datetime.utcnow().isoformat()
    batch.stats = stats
    db.commit()
    emit_ingestion_log(batch.id, "batch_queued", "Context graph rebuild queued", celery_task_id=task_id)


def mark_graph_rebuild_enqueue_failed(db: Session, batch: IngestionBatch, exc: Exception) -> None:
    message = f"Failed to enqueue context graph rebuild: {exception_message(exc)}"
    batch.status = "failed"
    batch.last_error = message
    batch.completed_at = datetime.utcnow()
    batch.stats = {**(batch.stats or {}), "phase": "failed", "enqueue_error": message, "manual_review_required": True}
    db.commit()
    emit_ingestion_log(batch.id, "batch_failed", message, error=message)


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
async def delete_knowledge_base(knowledge_base_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        knowledge_base = resolve_knowledge_base(
            db,
            knowledge_base_id,
            allow_deleting=True,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    knowledge_base_name = str(knowledge_base.name)
    try:
        recovery_owner = knowledge_base_delete_recovery_owner_token(
            db,
            knowledge_base.id,
        )
        async with knowledge_base_ingestion_resource_lock(
            db,
            knowledge_base.id,
            operation=KNOWLEDGE_BASE_DELETE_LOCK_OPERATION,
            batch_id=recovery_owner,
        ):
            stats = await run_bounded_source_io(
                delete_knowledge_base_data,
                db,
                knowledge_base,
            )
    except IngestionResourceBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ingestion_resource_busy",
                "message": str(exc),
                "resource_lock": exc.diagnostics,
            },
        ) from exc
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "deleted": True,
        "knowledge_base_id": knowledge_base_id,
        "knowledge_base_name": knowledge_base_name,
        "stats": stats,
    }


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
async def delete_knowledge_base_file(source_path: str, knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        recovery_owner = source_file_delete_lock_token(
            db,
            knowledge_base,
            source_path,
        )
        async with knowledge_base_ingestion_resource_lock(
            db,
            knowledge_base.id,
            operation=SOURCE_FILE_DELETE_LOCK_OPERATION,
            batch_id=recovery_owner,
        ):
            removed = await run_bounded_source_io(
                remove_knowledge_base_file,
                db,
                knowledge_base,
                source_path,
            )
    except UploadValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestionResourceBusyError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ingestion_resource_busy",
                "message": str(exc),
                "resource_lock": exc.diagnostics,
            },
        ) from exc
    except SourceFileDeleteError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "source_file_delete_recovery_pending",
                "message": str(exc),
            },
        ) from exc
    if not removed:
        raise HTTPException(status_code=404, detail="File not found")
    return {"removed": True}


@router.get("/knowledge_bases/{knowledge_base_id}/context-graph/stats")
def get_context_graph_stats(knowledge_base_id: str, db: Session = Depends(get_db)) -> dict:
    get_requested_knowledge_base(db, knowledge_base_id)
    return context_graph_stats(db, knowledge_base_id)


@router.post(
    "/knowledge_bases/{knowledge_base_id}/context-graph/freshness/reconcile",
    response_model=GraphFreshness,
)
def reconcile_context_graph_freshness_endpoint(
    knowledge_base_id: str,
    db: Session = Depends(get_db),
) -> dict:
    get_requested_knowledge_base(db, knowledge_base_id)
    payload = reconcile_context_graph_freshness(db, knowledge_base_id)
    db.commit()
    return payload


@router.get("/knowledge_bases/{knowledge_base_id}/graph/{graph_type}", response_model=GraphResponse)
def knowledge_base_layer_graph(knowledge_base_id: str, graph_type: str, limit: int = 200, db: Session = Depends(get_db)) -> dict:
    get_requested_knowledge_base(db, knowledge_base_id)
    try:
        payload = graph_layer_payload(
            db,
            knowledge_base_id,
            graph_type,
            limit=min(max(limit, 1), 1000),
        )
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/knowledge_bases/current/graph", response_model=GraphResponse)
@router.get("/knowledge_bases/default/graph", response_model=GraphResponse, include_in_schema=False)
def current_knowledge_base_graph(knowledge_base_id: str | None = None, graph_type: str = "chunk-relation", limit: int = 200, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        payload = graph_layer_payload(
            db,
            knowledge_base.id,
            graph_type,
            limit=min(max(limit, 1), 1000),
        )
        return payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/context-packages/{package_id}", response_model=ContextPackageResponse)
def context_package(package_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        payload = get_context_package(db, package_id)
    except ContextPackagePublicIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "context_package_replay_failed",
                "message": (
                    "Persisted context package is not replayable against the "
                    "current PostgreSQL evidence state."
                ),
                "failure_type": type(exc).__name__,
            },
        ) from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="Context package not found")
    return payload


@router.get("/retrieval-traces/{trace_id}/graph-steps", response_model=RetrievalTraceStepsResponse)
def retrieval_trace_steps(trace_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        payload = get_retrieval_trace_steps(db, trace_id)
    except RetrievalTraceAuditError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "retrieval_gray_zone_audit_failed",
                "message": str(exc),
                "audit": exc.audit,
            },
        ) from exc
    except (
        EntrySelectionTraceInvariantError,
        ContextPackagePublicIntegrityError,
    ) as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "retrieval_trace_replay_failed",
                "message": (
                    "Persisted retrieval trace is not replayable against the "
                    "current PostgreSQL evidence state."
                ),
                "failure_type": type(exc).__name__,
            },
        ) from exc
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
    batch = create_context_graph_rebuild_batch(db, knowledge_base.id, layers=[str(layer) for layer in request.layers])
    if get_settings().ingestion_execution_mode == "inline":
        summary = await run_context_graph_rebuild_batch(batch.id, execution_mode="inline")
        return {
            "batch_id": batch.id,
            "state": summary.get("state", "completed"),
            "mode": "four_layer_context_graph",
            "affected_documents": affected_documents,
            "dry_run": False,
            "stats": summary.get("stats", {}),
        }
    try:
        from worker_app.tasks import rebuild_context_graph_batch

        task = rebuild_context_graph_batch.apply_async(args=[batch.id], queue=get_settings().ingestion_task_queue)
        mark_graph_rebuild_enqueued(db, batch, getattr(task, "id", None))
    except Exception as exc:
        mark_graph_rebuild_enqueue_failed(db, batch, exc)
        raise HTTPException(status_code=503, detail=exception_message(exc)) from exc
    return {
        "batch_id": batch.id,
        "state": batch.status,
        "mode": "four_layer_context_graph",
        "affected_documents": affected_documents,
        "dry_run": False,
        "stats": batch.stats or {},
    }


@router.get("/knowledge-bases/{knowledge_base_id}/graph-operating-point/auto-tpe/latest", response_model=AutoTpeStatusResponse)
def get_auto_tpe_latest_status(knowledge_base_id: str, db: Session = Depends(get_db)) -> dict:
    get_requested_knowledge_base(db, knowledge_base_id)
    try:
        return latest_auto_tpe_status(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import get_db
from app.models import Chunk, IngestionBatch, IngestionJob
from app.schemas import BatchLogTokenResponse, BatchStartResponse, IngestionBatchSummary, JobStatusResponse, ParseUploadedFilesRequest, UploadFileResponse
from app.services.ingestion import (
    collect_source_documents,
    create_sync_batch,
    create_uploaded_files_batch,
    get_batch_status,
    get_job_status,
    register_uploaded_file,
    request_batch_cancel_control,
    resolve_knowledge_base,
    run_batch_ingestion,
    run_ingestion_job,
    run_uploaded_files_ingestion,
)
from app.services.ingestion_logs import (
    TERMINAL_LOG_EVENTS,
    create_log_stream_token,
    list_ingestion_logs,
    subscribe_ingestion_logs,
    unsubscribe_ingestion_logs,
    validate_log_stream_token,
)
from app.services.storage import save_upload

router = APIRouter()


def get_requested_knowledge_base(db: Session, knowledge_base_id: str | None = None):
    try:
        return resolve_knowledge_base(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def enqueue_error_message(exc: Exception) -> str:
    return f"Failed to enqueue ingestion task in celery mode: {type(exc).__name__}: {exc}"


async def enqueue_ingestion(job_id: str, source_path: str, trigger_source: str) -> None:
    settings = get_settings()
    if settings.ingestion_execution_mode == "inline":
        await run_ingestion_job(job_id, Path(source_path), trigger_source=trigger_source)
        return
    try:
        from worker_app.tasks import ingest_path

        ingest_path.apply_async(args=[source_path], kwargs={"trigger_source": trigger_source, "job_id": job_id}, queue=settings.ingestion_task_queue)
    except Exception as exc:
        mark_job_enqueue_failed(job_id, exc)
        raise


async def enqueue_batch(batch_id: str) -> None:
    settings = get_settings()
    if settings.ingestion_execution_mode == "inline":
        await run_batch_ingestion(batch_id)
        return
    try:
        from worker_app.tasks import ingest_batch

        task = ingest_batch.apply_async(args=[batch_id], queue=settings.ingestion_task_queue)
        mark_batch_enqueued(batch_id, "ingest_batch", getattr(task, "id", None))
    except Exception as exc:
        mark_batch_enqueue_failed(batch_id, exc)
        raise


async def enqueue_uploaded_batch(batch_id: str, file_paths: list[str], force: bool = False, full_reparse: bool = False) -> None:
    settings = get_settings()
    if settings.ingestion_execution_mode == "inline":
        await run_uploaded_files_ingestion(batch_id, file_paths, force=force, full_reparse=full_reparse, execution_mode="inline")
        return
    try:
        from worker_app.tasks import ingest_uploaded_batch

        task = ingest_uploaded_batch.apply_async(args=[batch_id, file_paths], kwargs={"force": force, "full_reparse": full_reparse}, queue=settings.ingestion_task_queue)
        mark_batch_enqueued(batch_id, "ingest_uploaded_batch", getattr(task, "id", None))
    except Exception as exc:
        mark_batch_enqueue_failed(batch_id, exc)
        raise


def mark_batch_enqueued(batch_id: str, task_name: str, task_id: str | None) -> None:
    from app.db import SessionLocal

    with SessionLocal() as session:
        batch = session.get(IngestionBatch, batch_id)
        if batch is None:
            return
        stats = dict(batch.stats or {})
        stats["ingestion_execution_mode"] = "celery"
        stats["celery_task_name"] = task_name
        if task_id:
            stats["celery_task_id"] = task_id
            stats["batch_task_ids"] = sorted({*stats.get("batch_task_ids", []), str(task_id)})
            stats["task_last_seen_at"] = datetime.utcnow().isoformat()
        batch.stats = stats
        session.commit()


def mark_batch_enqueue_failed(batch_id: str, exc: Exception) -> None:
    from app.db import SessionLocal
    from app.services.ingestion_logs import emit_ingestion_log

    message = enqueue_error_message(exc)
    with SessionLocal() as session:
        batch = session.get(IngestionBatch, batch_id)
        if batch is None:
            return
        batch.status = "failed"
        batch.last_error = message
        batch.completed_at = datetime.utcnow()
        batch.stats = {**(batch.stats or {}), "ingestion_execution_mode": "celery", "enqueue_error": message}
        session.commit()
        emit_ingestion_log(batch_id, "batch_failed", message, error=message, ingestion_execution_mode="celery")


def mark_job_enqueue_failed(job_id: str, exc: Exception) -> None:
    from app.db import SessionLocal

    message = enqueue_error_message(exc)
    with SessionLocal() as session:
        job = session.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = "failed"
        job.error_message = message
        job.stats = {**(job.stats or {}), "ingestion_execution_mode": "celery", "enqueue_error": message}
        session.commit()


def enqueue_batch_background(batch_id: str) -> None:
    asyncio.run(enqueue_batch(batch_id))


def enqueue_uploaded_batch_background(batch_id: str, file_paths: list[str], force: bool = False, full_reparse: bool = False) -> None:
    asyncio.run(enqueue_uploaded_batch(batch_id, file_paths, force=force, full_reparse=full_reparse))


@router.post("/files/upload", response_model=UploadFileResponse)
async def upload_file(
    knowledge_base_id: str | None = None,
    upload: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    stored_path = await save_upload(upload, knowledge_base.name)
    try:
        document, job = register_uploaded_file(db, knowledge_base, stored_path)
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    return {"document_id": document.id, "job_id": job.id, "status": "queued", "source_path": str(stored_path)}


@router.post("/ingestion/parse-uploaded-files", response_model=BatchStartResponse)
async def parse_uploaded_files(
    request: ParseUploadedFilesRequest,
    background_tasks: BackgroundTasks,
    knowledge_base_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    storage_root = Path(get_settings().knowledge_base_paths_for_name(knowledge_base.name)["storage_root"]).resolve()
    requested_paths = request.file_paths or [str(path) for path in collect_source_documents(storage_root)]
    if not requested_paths:
        raise HTTPException(status_code=400, detail="No files found in knowledge base storage")
    file_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for raw_path in requested_paths:
        path = Path(raw_path).resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        if path != storage_root and storage_root not in path.parents:
            raise HTTPException(status_code=400, detail=f"File is outside knowledge base storage: {path}")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        file_paths.append(path)
    if request.full_reparse:
        active_chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.knowledge_base_id == knowledge_base.id, Chunk.state == "active")) or 0
        if active_chunks <= 0:
            raise HTTPException(status_code=409, detail="Full reparse is unavailable before the first successful parse")
    batch = create_uploaded_files_batch(db, knowledge_base.id, file_paths, force=request.force, full_reparse=request.full_reparse)
    serialized_paths = [str(path) for path in file_paths]
    if get_settings().ingestion_execution_mode == "celery":
        try:
            await enqueue_uploaded_batch(batch.id, serialized_paths, request.force, request.full_reparse)
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "ingestion_enqueue_failed", "message": enqueue_error_message(exc)}) from exc
    else:
        background_tasks.add_task(enqueue_uploaded_batch_background, batch.id, serialized_paths, request.force, request.full_reparse)
    return {"batch_id": batch.id, "state": "queued"}


@router.post("/ingestion/parse-storage", response_model=BatchStartResponse)
async def parse_storage_directory(background_tasks: BackgroundTasks, knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    root = Path(get_settings().knowledge_base_paths_for_name(knowledge_base.name)["storage_root"])
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Storage root not found: {root}")
    batch = create_sync_batch(db, knowledge_base.id, root, trigger_source="storage")
    if get_settings().ingestion_execution_mode == "celery":
        try:
            await enqueue_batch(batch.id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "ingestion_enqueue_failed", "message": enqueue_error_message(exc)}) from exc
    else:
        background_tasks.add_task(enqueue_batch_background, batch.id)
    return {"batch_id": batch.id, "state": "queued"}


@router.get("/ingestion/batches/{batch_id}", response_model=IngestionBatchSummary)
def batch_status(batch_id: str, db: Session = Depends(get_db)) -> dict:
    batch = get_batch_status(db, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


@router.post("/ingestion/batches/{batch_id}/cancel", response_model=IngestionBatchSummary)
async def cancel_batch(batch_id: str, knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    try:
        batch = request_batch_cancel_control(db, batch_id, knowledge_base.id)
    except PermissionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if batch is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return batch


@router.post("/ingestion/batches/{batch_id}/log-token", response_model=BatchLogTokenResponse)
def batch_log_token(batch_id: str, db: Session = Depends(get_db)) -> dict:
    if get_batch_status(db, batch_id) is None:
        raise HTTPException(status_code=404, detail="Batch not found")
    return create_log_stream_token(batch_id)


@router.get("/ingestion/batches/{batch_id}/logs")
async def batch_logs(batch_id: str, token: str | None = None, x_api_key: str | None = Header(default=None)):
    allowed_keys = get_settings().api_key_list
    header_authorized = bool(x_api_key and x_api_key in allowed_keys)
    if allowed_keys and not header_authorized:
        try:
            validate_log_stream_token(batch_id, token)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
    elif token:
        try:
            validate_log_stream_token(batch_id, token)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    from app.db import SessionLocal

    with SessionLocal() as session:
        batch_exists = get_batch_status(session, batch_id) is not None
    if not batch_exists:
        async def missing_stream():
            yield f"data: {json.dumps({'timestamp': datetime.utcnow().isoformat(), 'event': 'batch_missing', 'message': 'Batch no longer exists.', 'state': 'missing'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(missing_stream(), media_type="text/event-stream")

    async def event_stream():
        emitted: set[str] = set()

        def event_key(item: dict) -> str:
            return str(item.get("log_id") or item.get("synthetic_key") or f"{item.get('timestamp')}:{item.get('event')}:{item.get('message')}")

        def format_new(item: dict) -> str | None:
            key = event_key(item)
            if key in emitted:
                return None
            emitted.add(key)
            return f"data: {json.dumps(item, ensure_ascii=False, default=str)}\n\n"

        def batch_snapshot_event() -> dict | None:
            with SessionLocal() as session:
                snapshot = get_batch_status(session, batch_id)
            if snapshot is None:
                return None
            state = snapshot["state"]
            terminal_events = {
                "completed": "batch_completed",
                "failed": "batch_failed",
                "partial_failed": "batch_partial_failed",
                "skipped": "batch_skipped",
                "cancelled": "batch_cancelled",
            }
            return {
                "synthetic_key": f"snapshot:{state}:{snapshot['processed_files']}:{snapshot['success_count']}:{snapshot['failure_count']}:{snapshot['skipped_count']}",
                "timestamp": datetime.utcnow().isoformat(),
                "event": terminal_events.get(state, "batch_status"),
                "message": f"Batch {state}: {snapshot['processed_files']}/{snapshot['total_files']} processed",
                "state": state,
                "processed_files": snapshot["processed_files"],
                "total_files": snapshot["total_files"],
                "success_count": snapshot["success_count"],
                "failure_count": snapshot["failure_count"],
                "skipped_count": snapshot["skipped_count"],
            }

        history, subscriber = subscribe_ingestion_logs(batch_id)
        try:
            for item in history:
                chunk = format_new(item)
                if chunk:
                    yield chunk
                if item.get("event") in TERMINAL_LOG_EVENTS:
                    return
            while True:
                for item in list_ingestion_logs(batch_id):
                    chunk = format_new(item)
                    if chunk:
                        yield chunk
                    if item.get("event") in TERMINAL_LOG_EVENTS:
                        return
                snapshot = batch_snapshot_event()
                if snapshot:
                    chunk = format_new(snapshot)
                    if chunk:
                        yield chunk
                    if snapshot.get("event") in TERMINAL_LOG_EVENTS:
                        return
                try:
                    item = await asyncio.to_thread(subscriber.get, True, 2)
                except Exception:
                    yield ": heartbeat\n\n"
                    continue
                chunk = format_new(item)
                if chunk:
                    yield chunk
                if item.get("event") in TERMINAL_LOG_EVENTS:
                    return
        finally:
            unsubscribe_ingestion_logs(batch_id, subscriber)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, db: Session = Depends(get_db)) -> dict:
    job = get_job_status(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

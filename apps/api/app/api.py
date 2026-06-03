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
from app.models import AgentRun, AgentTraceEvent, Concept, ConceptRelation, Document, IngestionBatch, IngestionJob, QASession
from app.core.config import get_settings
from app.schemas import (
    AgentRequest,
    AgentResponse,
    BatchLogTokenResponse,
    BatchStartResponse,
    CleanupStaleDataResponse,
    CleanupStaleGraphResponse,
    ConceptCard,
    CourseFileSummary,
    CourseCreateRequest,
    CourseSummary,
    DashboardSnapshot,
    DeleteCourseResponse,
    DeleteResponse,
    GraphNodeDetail,
    RebuildGraphResponse,
    GraphResponse,
    IngestionBatchSummary,
    JobStatusResponse,
    ModelSettingsResponse,
    ModelSettingsUpdate,
    ParseUploadedFilesRequest,
    QARequest,
    QAResponse,
    RebuildGraphRequest,
    RefreshResponse,
    RuntimeCheckResponse,
    QuerySemanticGraphRequest,
    SearchRequest,
    SearchResponse,
    SessionMessagesResponse,
    SessionSummary,
    TaskStatusResponse,
    UploadFileResponse,
)
from app.services.concept_graph import get_concept_cards, get_graph_node_detail, get_graph_payload, get_query_semantic_graph_payload, rebuild_course_graph, incremental_update_course_graph
from app.services.embeddings import is_degraded_mode
from app.services.agent_graph import run_agent, run_to_task_status, stream_agent_events
from app.services.ingestion import (
    create_course_space,
    collect_source_documents,
    create_uploaded_files_batch,
    create_job,
    create_sync_batch,
    get_batch_status,
    list_course_summaries,
    request_batch_cancel,
    register_uploaded_file,
    resolve_course,
    run_batch_ingestion,
    run_graph_rebuild_background,
    run_ingestion_job,
    run_uploaded_files_ingestion,
    remove_course_file,
    summarize_course,
)
from app.services.ingestion_logs import (
    TERMINAL_LOG_EVENTS,
    create_log_stream_token,
    list_ingestion_logs,
    subscribe_ingestion_logs,
    unsubscribe_ingestion_logs,
    validate_log_stream_token,
)
from app.services.maintenance import MaintenanceConflict, cleanup_stale_data, cleanup_stale_graph, delete_course_data
from app.services.retrieval import (
    evidence_first_search_chunks_with_audit,
    get_dashboard_snapshot,
    get_job_status,
    list_course_files,
    search_chunks_with_audit,
)
from app.services.runtime_settings import model_settings_payload, normalize_env_file, runtime_check_payload, update_model_settings
from app.services.storage import save_upload

router = APIRouter()


def get_requested_course(db: Session, course_id: str | None = None):
    try:
        return resolve_course(db, course_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "degraded_mode": is_degraded_mode()}


@router.get("/settings/model", response_model=ModelSettingsResponse)
def get_model_settings() -> dict:
    return model_settings_payload()


@router.put("/settings/model", response_model=ModelSettingsResponse)
def save_model_settings(request: ModelSettingsUpdate) -> dict:
    normalize_env_file()
    return update_model_settings(request.model_dump(exclude_unset=True))


@router.get("/settings/runtime-check", response_model=RuntimeCheckResponse)
def get_runtime_check() -> dict:
    return runtime_check_payload()


@router.get("/courses", response_model=list[CourseSummary])
def list_courses(db: Session = Depends(get_db)) -> list[dict]:
    return list_course_summaries(db)


@router.post("/courses", response_model=CourseSummary)
def create_course(request: CourseCreateRequest, db: Session = Depends(get_db)) -> dict:
    try:
        course = create_course_space(db, request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summarize_course(db, course)


@router.delete("/courses/{course_id}", response_model=DeleteCourseResponse)
def delete_course(course_id: str, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    try:
        stats = delete_course_data(db, course)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": True, **stats}


@router.get("/courses/current/dashboard", response_model=DashboardSnapshot)
@router.get("/courses/default/dashboard", response_model=DashboardSnapshot, include_in_schema=False)
def course_dashboard(course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    return get_dashboard_snapshot(db, course.id)


@router.post("/courses/current/refresh", response_model=RefreshResponse)
def refresh_current_course(course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    return {"course_id": course.id, "refreshed_at": datetime.utcnow()}


@router.get("/course-files", response_model=list[CourseFileSummary])
def course_files(course_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    course = get_requested_course(db, course_id)
    return list_course_files(db, course.id)


@router.delete("/course-files")
def delete_course_file(source_path: str, course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    if not remove_course_file(db, course, source_path):
        raise HTTPException(status_code=404, detail="File not found")
    return {"removed": True}


@router.post("/maintenance/cleanup-stale-data", response_model=CleanupStaleDataResponse)
def cleanup_course_stale_data(course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    try:
        return cleanup_stale_data(db, course.id, course.name)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/maintenance/cleanup-stale-graph", response_model=CleanupStaleGraphResponse)
def cleanup_course_stale_graph(course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    try:
        return cleanup_stale_graph(db, course.id)
    except MaintenanceConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/maintenance/rebuild-graph", response_model=RebuildGraphResponse)
async def rebuild_graph_endpoint(
    background_tasks: BackgroundTasks,
    request: RebuildGraphRequest = RebuildGraphRequest(),
    course_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    from app.services.ingestion import active_batch_for_course

    course = get_requested_course(db, course_id)
    if request.mode not in {"incremental", "full"}:
        raise HTTPException(status_code=400, detail="mode must be incremental or full")
    if request.mode == "full" and not request.dry_run and not request.confirm_destructive:
        raise HTTPException(status_code=400, detail="confirm_destructive is required for full semantic KG rebuild")
    affected_documents = db.scalar(select(func.count(Document.id)).where(Document.course_id == course.id)) or 0
    semantic_entities = db.scalar(select(func.count(Concept.id)).where(Concept.course_id == course.id)) or 0
    semantic_relations = db.scalar(select(func.count(ConceptRelation.id)).where(ConceptRelation.course_id == course.id)) or 0
    if request.dry_run:
        return {
            "batch_id": None,
            "state": "dry_run",
            "mode": request.mode,
            "affected_documents": affected_documents,
            "dry_run": True,
            "semantic_entities": semantic_entities,
            "semantic_relations": semantic_relations,
        }
    active = active_batch_for_course(db, course.id)
    if active is not None:
        raise HTTPException(status_code=409, detail=f"课程已有活跃批次：{active.id}")
    batch = IngestionBatch(
        course_id=course.id,
        source_root="graph_rebuild",
        trigger_source="rebuild_graph",
        status="extracting_graph",
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    if get_settings().ingestion_execution_mode == "celery":
        try:
            enqueue_graph_rebuild_batch(batch.id, course.id, request.mode)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "graph_rebuild_enqueue_failed", "message": enqueue_error_message(exc)},
            ) from exc
    else:
        background_tasks.add_task(run_graph_rebuild_background, batch.id, course.id, request.mode)
    return {
        "batch_id": batch.id,
        "state": "extracting_graph",
        "mode": request.mode,
        "affected_documents": affected_documents,
        "dry_run": False,
        "semantic_entities": semantic_entities,
        "semantic_relations": semantic_relations,
    }


@router.get("/courses/current/graph", response_model=GraphResponse)
@router.get("/courses/default/graph", response_model=GraphResponse, include_in_schema=False)
def course_graph(course_id: str | None = None, graph_type: str | None = None, db: Session = Depends(get_db)) -> dict:
    if graph_type not in {"semantic", "structural", "evidence"}:
        raise HTTPException(status_code=400, detail="graph_type is required and must be one of: semantic, structural, evidence")
    course = get_requested_course(db, course_id)
    return get_graph_payload(db, course.id, graph_type=graph_type)


@router.get("/graph/chapters/{chapter}", response_model=GraphResponse)
def chapter_graph(chapter: str, course_id: str | None = None, graph_type: str | None = None, db: Session = Depends(get_db)) -> dict:
    if graph_type not in {"semantic", "structural", "evidence"}:
        raise HTTPException(status_code=400, detail="graph_type is required and must be one of: semantic, structural, evidence")
    course = get_requested_course(db, course_id)
    return get_graph_payload(db, course.id, chapter=chapter if graph_type in {"semantic", "evidence"} else None, graph_type=graph_type)


@router.get("/graph/nodes/{concept_id}", response_model=GraphNodeDetail)
def graph_node_detail(concept_id: str, course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    payload = get_graph_node_detail(db, course.id, concept_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Concept node not found")
    return payload


@router.post("/search/semantic-graph", response_model=GraphResponse)
def search_semantic_graph(request: QuerySemanticGraphRequest, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, request.course_id)
    return get_query_semantic_graph_payload(db, course.id, request.chunk_ids, query=request.query)


@router.get("/concepts", response_model=list[ConceptCard])
def concept_cards(course_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    course = get_requested_course(db, course_id)
    return get_concept_cards(db, course.id)


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


async def enqueue_uploaded_batch(batch_id: str, file_paths: list[str], force: bool = False, rebuild_graph_mode: str = "none") -> None:
    settings = get_settings()
    if settings.ingestion_execution_mode == "inline":
        await run_uploaded_files_ingestion(batch_id, file_paths, force=force, rebuild_graph_mode=rebuild_graph_mode)
        return
    try:
        from worker_app.tasks import ingest_uploaded_batch

        task = ingest_uploaded_batch.apply_async(args=[batch_id, file_paths], kwargs={"force": force, "rebuild_graph_mode": rebuild_graph_mode}, queue=settings.ingestion_task_queue)
        mark_batch_enqueued(batch_id, "ingest_uploaded_batch", getattr(task, "id", None))
    except Exception as exc:
        mark_batch_enqueue_failed(batch_id, exc)
        raise


def enqueue_error_message(exc: Exception) -> str:
    return f"Failed to enqueue ingestion task in celery mode: {type(exc).__name__}: {exc}"


def mark_batch_enqueued(batch_id: str, task_name: str, task_id: str | None) -> None:
    from app.db import SessionLocal

    session = SessionLocal()
    try:
        batch = session.get(IngestionBatch, batch_id)
        if batch is None:
            return
        stats = dict(batch.stats or {})
        stats["ingestion_execution_mode"] = "celery"
        stats["celery_task_name"] = task_name
        if task_id:
            stats["celery_task_id"] = task_id
        batch.stats = stats
        session.commit()
    finally:
        session.close()


def mark_batch_enqueue_failed(batch_id: str, exc: Exception) -> None:
    from app.db import SessionLocal
    from app.services.ingestion_logs import emit_ingestion_log

    message = enqueue_error_message(exc)
    session = SessionLocal()
    try:
        batch = session.get(IngestionBatch, batch_id)
        if batch is None:
            return
        batch.status = "failed"
        batch.last_error = message
        batch.completed_at = datetime.utcnow()
        batch.stats = {**(batch.stats or {}), "ingestion_execution_mode": "celery", "enqueue_error": message}
        session.commit()
        emit_ingestion_log(batch_id, "batch_failed", message, error=message, ingestion_execution_mode="celery")
    finally:
        session.close()


def mark_job_enqueue_failed(job_id: str, exc: Exception) -> None:
    from app.db import SessionLocal

    message = enqueue_error_message(exc)
    session = SessionLocal()
    try:
        job = session.get(IngestionJob, job_id)
        if job is None:
            return
        job.status = "failed"
        job.error_message = message
        job.stats = {**(job.stats or {}), "ingestion_execution_mode": "celery", "enqueue_error": message}
        session.commit()
    finally:
        session.close()


def enqueue_graph_rebuild_batch(batch_id: str, course_id: str, mode: str) -> None:
    settings = get_settings()
    try:
        from worker_app.tasks import rebuild_course_graph_task

        task = rebuild_course_graph_task.apply_async(args=[batch_id, course_id, mode], queue=settings.ingestion_task_queue)
        mark_batch_enqueued(batch_id, "rebuild_course_graph", getattr(task, "id", None))
    except Exception as exc:
        mark_batch_enqueue_failed(batch_id, exc)
        raise


def enqueue_batch_background(batch_id: str) -> None:
    asyncio.run(enqueue_batch(batch_id))


def enqueue_uploaded_batch_background(batch_id: str, file_paths: list[str], force: bool = False, rebuild_graph_mode: str = "none") -> None:
    asyncio.run(enqueue_uploaded_batch(batch_id, file_paths, force=force, rebuild_graph_mode=rebuild_graph_mode))


@router.post("/files/upload", response_model=UploadFileResponse)
async def upload_file(
    course_id: str | None = None,
    upload: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    course = get_requested_course(db, course_id)
    stored_path = await save_upload(upload, course.name)
    try:
        document, job = register_uploaded_file(db, course, stored_path)
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    return {"document_id": document.id, "job_id": job.id, "status": "queued", "source_path": str(stored_path)}


@router.post("/ingestion/parse-uploaded-files", response_model=BatchStartResponse)
async def parse_uploaded_files(
    request: ParseUploadedFilesRequest,
    background_tasks: BackgroundTasks,
    course_id: str | None = None,
    db: Session = Depends(get_db),
) -> dict:
    course = get_requested_course(db, course_id)
    if request.rebuild_graph_mode == "full" and not request.confirm_destructive_graph_rebuild:
        raise HTTPException(status_code=400, detail="confirm_destructive_graph_rebuild is required for full graph rebuild after parsing")
    storage_root = Path(get_settings().course_paths_for_name(course.name)["storage_root"]).resolve()
    requested_paths = request.file_paths or [str(path) for path in collect_source_documents(storage_root)]
    if not requested_paths:
        raise HTTPException(status_code=400, detail="No files found in course storage")
    file_paths = []
    seen_paths: set[Path] = set()
    for raw_path in requested_paths:
        path = Path(raw_path).resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        if storage_root not in path.parents and path != storage_root:
            raise HTTPException(status_code=400, detail=f"File is outside course storage: {path}")
        if path in seen_paths:
            continue
        seen_paths.add(path)
        file_paths.append(path)
    batch = create_uploaded_files_batch(db, course.id, file_paths, force=request.force, rebuild_graph_mode=request.rebuild_graph_mode)
    serialized_paths = [str(path) for path in file_paths]
    if get_settings().ingestion_execution_mode == "celery":
        try:
            await enqueue_uploaded_batch(batch.id, serialized_paths, request.force, request.rebuild_graph_mode)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "ingestion_enqueue_failed", "message": enqueue_error_message(exc)},
            ) from exc
    else:
        background_tasks.add_task(enqueue_uploaded_batch_background, batch.id, serialized_paths, request.force, request.rebuild_graph_mode)
    return {"batch_id": batch.id, "state": "queued"}


@router.post("/ingestion/parse-storage", response_model=BatchStartResponse)
@router.post("/ingestion/sync-source", response_model=BatchStartResponse, include_in_schema=False)
async def parse_storage_directory(background_tasks: BackgroundTasks, course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    root = Path(get_settings().course_paths_for_name(course.name)["storage_root"])
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Storage root not found: {root}")
    batch = create_sync_batch(db, course.id, root, trigger_source="storage")
    if get_settings().ingestion_execution_mode == "celery":
        try:
            await enqueue_batch(batch.id)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "ingestion_enqueue_failed", "message": enqueue_error_message(exc)},
            ) from exc
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
def cancel_batch(batch_id: str, course_id: str | None = None, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, course_id)
    try:
        batch = request_batch_cancel(db, batch_id, course.id)
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
    from app.db import SessionLocal

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

    with SessionLocal() as session:
        batch_exists = get_batch_status(session, batch_id) is not None

    if not batch_exists:
        async def missing_stream():
            yield f"data: {json.dumps({'timestamp': datetime.utcnow().isoformat(), 'event': 'batch_missing', 'message': 'Batch no longer exists. The local UI state can be cleared.', 'state': 'missing'}, ensure_ascii=False)}\n\n"

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
            return f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        def batch_snapshot_event() -> dict | None:
            from app.db import SessionLocal

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
            }
            event = terminal_events.get(state, "batch_status")
            return {
                "synthetic_key": f"snapshot:{state}:{snapshot['processed_files']}:{snapshot['success_count']}:{snapshot['failure_count']}:{snapshot['skipped_count']}",
                "timestamp": datetime.utcnow().isoformat(),
                "event": event,
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
                latest = list_ingestion_logs(batch_id)
                for item in latest:
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


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, request.course_id)
    try:
        results, model_audit = await search_chunks_with_audit(db, course.id, request.query, request.filters, request.top_k)
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        raise HTTPException(
            status_code=502,
            detail={
                "code": "search_embedding_failed",
                "title": "Search embedding request failed",
                "message": (
                    "The search query could not be embedded by the configured model API. "
                    "Retrieval did not fall back to fake or lexical-only results."
                ),
                "issues": [
                    {
                        "code": "embedding_api_unreachable",
                        "title": "Embedding API is unreachable from the API container",
                        "message": message,
                        "fix_commands": [
                            "Check EMBEDDING_BASE_URL and EMBEDDING_RESOLVE_IP in .env.",
                            "Verify the API container can reach the embedding endpoint.",
                        ],
                    }
                ],
                "fix_commands": [
                    "docker logs --tail 120 course-kg-api",
                    "docker exec course-kg-api curl -I https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                ],
            },
        ) from exc
    return {"query": request.query, "results": results, "degraded_mode": is_degraded_mode(), "model_audit": model_audit}


@router.post("/search/graph-enhanced", response_model=SearchResponse)
async def graph_search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    course = get_requested_course(db, request.course_id)
    try:
        results, audit = await evidence_first_search_chunks_with_audit(
            db,
            course.id,
            request.query,
            request.filters,
            request.top_k,
            route="multi_hop_research",
        )
    except Exception as exc:
        message = str(exc) or type(exc).__name__
        raise HTTPException(status_code=502, detail={"code": "graph_search_failed", "message": message}) from exc
    model_audit = audit
    return {"query": request.query, "results": results, "degraded_mode": is_degraded_mode(), "model_audit": model_audit}


@router.post("/qa", response_model=QAResponse)
async def qa(request: QARequest, db: Session = Depends(get_db)) -> dict:
    get_requested_course(db, request.course_id)
    agent_request = AgentRequest(
        question=request.question,
        session_id=request.session_id,
        course_id=request.course_id,
        filters=request.filters,
        top_k=request.top_k,
        history=request.history,
        stream_trace=False,
    )
    return await run_agent(db, agent_request)


@router.post("/qa/stream")
async def qa_stream(request: QARequest) -> StreamingResponse:
    from app.db import SessionLocal
    with SessionLocal() as db:
        get_requested_course(db, request.course_id)
    agent_request = AgentRequest(
        question=request.question,
        session_id=request.session_id,
        course_id=request.course_id,
        filters=request.filters,
        top_k=request.top_k,
        history=request.history,
        stream_trace=True,
    )

    async def event_stream():
        from app.db import SessionLocal
        db = SessionLocal()
        try:
            async for event in stream_agent_events(db, agent_request):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            db.close()
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/agent", response_model=AgentResponse)
async def agent_call(request: AgentRequest, db: Session = Depends(get_db)) -> dict:
    get_requested_course(db, request.course_id)
    return await run_agent(db, request)


@router.get("/agent/runs/{run_id}", response_model=TaskStatusResponse)
@router.get("/tasks/{run_id}", response_model=TaskStatusResponse)
def agent_run_status(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run_to_task_status(run)


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(course_id: str | None = None, db: Session = Depends(get_db)) -> list[QASession]:
    course = get_requested_course(db, course_id)
    return list(
        db.scalars(
            select(QASession).where(QASession.course_id == course.id).order_by(QASession.updated_at.desc())
        ).all()
    )


@router.get("/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: str, db: Session = Depends(get_db)) -> QASession:
    session = db.get(QASession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
def get_session_messages(session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.get(QASession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session.id, "messages": session.transcript or []}


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.get(QASession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    run_ids = [run.id for run in db.scalars(select(AgentRun).where(AgentRun.session_id == session_id)).all()]
    if run_ids:
        db.query(AgentTraceEvent).filter(AgentTraceEvent.run_id.in_(run_ids)).delete(synchronize_session=False)
        db.query(AgentRun).filter(AgentRun.id.in_(run_ids)).delete(synchronize_session=False)
    db.delete(session)
    db.commit()
    return {"deleted": True}

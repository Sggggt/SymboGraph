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
from app.models import ActiveChunk, AgentRun, AgentTraceEvent, Document, EvidenceAtom, EvidenceEdge, EvidenceGraphState, IngestionBatch, IngestionJob, QASession
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
    TaskStatusResponse,
    UploadFileResponse,
)
from app.services.evidence_graph_payload import get_graph_payload, get_query_evidence_graph_payload
from app.services.evidence_graph import publish_global_evidence_graph_state
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
from app.services.maintenance import MaintenanceConflict, cleanup_stale_data, delete_knowledge_base_data
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



@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, request.knowledge_base_id)
    try:
        results, model_audit = await search_chunks_with_audit(db, KnowledgeBase.id, request.query, request.filters, request.top_k)
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
                    "docker logs --tail 120 KnowledgeBase-kg-api",
                    "docker exec KnowledgeBase-kg-api curl -I https://dashscope.aliyuncs.com/compatible-mode/v1/models",
                ],
            },
        ) from exc
    return {"query": request.query, "results": results, "degraded_mode": is_degraded_mode(), "model_audit": model_audit}


@router.post("/search/graph-enhanced", response_model=SearchResponse)
async def graph_search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    KnowledgeBase = get_requested_knowledge_base(db, request.knowledge_base_id)
    try:
        results, audit = await evidence_first_search_chunks_with_audit(
            db,
            KnowledgeBase.id,
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
    get_requested_knowledge_base(db, request.knowledge_base_id)
    agent_request = AgentRequest(
        question=request.question,
        session_id=request.session_id,
        knowledge_base_id=request.knowledge_base_id,
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
        get_requested_knowledge_base(db, request.knowledge_base_id)
    agent_request = AgentRequest(
        question=request.question,
        session_id=request.session_id,
        knowledge_base_id=request.knowledge_base_id,
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
    get_requested_knowledge_base(db, request.knowledge_base_id)
    return await run_agent(db, request)


@router.get("/agent/runs/{run_id}", response_model=TaskStatusResponse)
@router.get("/tasks/{run_id}", response_model=TaskStatusResponse)
def agent_run_status(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return run_to_task_status(run)

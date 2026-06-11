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

@router.get("/settings/model", response_model=ModelSettingsResponse)
def get_model_settings() -> dict:
    return model_settings_payload()


@router.put("/settings/model", response_model=ModelSettingsResponse)
def save_model_settings(request: ModelSettingsUpdate) -> dict:
    return update_model_settings(request.model_dump(exclude_unset=True))


@router.get("/settings/runtime-check", response_model=RuntimeCheckResponse)
def get_runtime_check() -> dict:
    return runtime_check_payload()


@router.get("/settings/profiles", response_model=list[StrategyProfileSummary])
def get_strategy_profiles(db: Session = Depends(get_db)) -> list[dict]:
    return list_profiles(db)


@router.get("/settings/profiles/{profile_id}", response_model=StrategyProfileDetail)
def get_strategy_profile(profile_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return profile_to_payload(get_profile_or_raise(db, profile_id))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/settings/profiles", response_model=StrategyProfileMutationResponse)
def create_strategy_profile(request: StrategyProfileCreateRequest, db: Session = Depends(get_db)) -> dict:
    try:
        profile, warnings = create_profile(
            db,
            name=request.name,
            library_type=request.library_type,
            profile_json=request.profile_json,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"profile": profile_to_payload(profile), "warnings": warnings}


@router.put("/settings/profiles/{profile_id}", response_model=StrategyProfileMutationResponse)
def update_strategy_profile(profile_id: str, request: StrategyProfileUpdateRequest, db: Session = Depends(get_db)) -> dict:
    try:
        profile, warnings = update_profile(
            db,
            profile_id,
            name=request.name,
            library_type=request.library_type,
            profile_json=request.profile_json,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"profile": profile_to_payload(profile), "warnings": warnings}


@router.post("/settings/profiles/{profile_id}/copy", response_model=StrategyProfileDetail)
def copy_strategy_profile(profile_id: str, request: StrategyProfileCopyRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return profile_to_payload(copy_profile(db, profile_id, name=request.name))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/settings/profiles/{profile_id}", response_model=DeleteResponse)
def delete_strategy_profile(profile_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        delete_profile(db, profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": True}


@router.post("/settings/profiles/bind", response_model=KnowledgeBaseSummary)
def bind_strategy_profile(request: StrategyProfileBindRequest, db: Session = Depends(get_db)) -> dict:
    try:
        KnowledgeBase = bind_profile_to_knowledge_base(db, knowledge_base_id=request.knowledge_base_id, profile_id=request.profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summarize_knowledge_base(db, KnowledgeBase)


@router.post("/settings/profile-drafts", response_model=StrategyProfileDraftResponse)
async def draft_strategy_profile(request: StrategyProfileDraftRequest, db: Session = Depends(get_db)) -> dict:
    base = request.base_profile_json
    if base is None and request.base_profile_id:
        try:
            base = get_profile_or_raise(db, request.base_profile_id).profile_json
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        return await generate_profile_draft(request.prompt, base_profile=base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/settings/profile-assistant/{session_id}", response_model=StrategyProfileAssistantStateResponse)
def get_strategy_profile_assistant_state(session_id: str) -> dict:
    return get_profile_assistant_state(session_id)


@router.post("/settings/profile-assistant/stream")
async def stream_strategy_profile_assistant(request: StrategyProfileAssistantRequest, db: Session = Depends(get_db)):
    base = request.base_profile_json
    if base is None and request.base_profile_id:
        try:
            base = get_profile_or_raise(db, request.base_profile_id).profile_json
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def event_stream():
        try:
            async for event in stream_profile_assistant_events(
                prompt=request.prompt,
                session_id=request.session_id,
                base_profile_id=request.base_profile_id,
                base_profile=base,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)}, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

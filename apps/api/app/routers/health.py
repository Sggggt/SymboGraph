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

@router.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "degraded_mode": is_degraded_mode()}

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AgentRun, AgentTraceEvent
from app.services.agent_graph import trace_event_to_payload
from app.schemas import (
    ActiveContextGraphAdmissionErrorDetail,
    ActiveContextGraphAdmissionErrorResponse,
    ActiveContextGraphAdmissionIssue,
    AgentPEAuditResponse,
    AgentRequest,
    AgentResponse,
    QARequest,
    QAResponse,
    SearchRequest,
    SearchResponse,
    TaskStatusResponse,
    public_search_result_payload,
)
from app.services.agent_graph import cancel_agent_run, run_agent, run_to_task_status, stream_agent_events
from app.services.agent_admission import AgentAdmissionError, acquire_agent_request_slot
from app.services.qa_performance import QAPerformance, qa_stage
from app.services.agent_pe_audit import (
    AgentPEAuditIntegrityError,
    load_agent_pe_audit,
)
from app.services.context_graph import (
    ACTIVE_CONTEXT_GRAPH_ADMISSION_PROTOCOL_VERSION,
    ActiveContextGraphAdmissionError,
    GrayZoneTraceInvariantError,
    active_graph_admission_gate,
)
from app.services.conversation_state import (
    ConversationStateConflictError,
    ConversationStateIntegrityError,
    ConversationStateNotFoundError,
    load_conversation_state,
    merge_search_filters_with_conversation_constraints,
)
from app.services.embeddings import is_degraded_mode
from app.services.error_sanitizer import external_error_payload, public_exception_message
from app.services.ingestion import resolve_knowledge_base
from app.services.retrieval import layered_context_search_chunks_with_audit, search_chunks_with_audit

router = APIRouter()
SSE_HEARTBEAT_SECONDS = 10.0
ACTIVE_CONTEXT_GRAPH_ADMISSION_STATUS_CODE = 409
ACTIVE_CONTEXT_GRAPH_REBUILD_FIX_COMMANDS = (
    "Run a full contextual-index and context-graph rebuild for this knowledge base.",
    "Retry retrieval only after graph freshness reports is_stale=false.",
)
ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES = {
    ACTIVE_CONTEXT_GRAPH_ADMISSION_STATUS_CODE: {
        "model": ActiveContextGraphAdmissionErrorResponse,
        "description": "The active contextual index or a dependent graph layer requires rebuild.",
    }
}


async def stream_sse_frames(
    events: AsyncIterator[dict],
    *,
    heartbeat_seconds: float = SSE_HEARTBEAT_SECONDS,
) -> AsyncIterator[str]:
    """Frame one agent stream while keeping idle HTTP connections alive."""

    interval = max(0.01, float(heartbeat_seconds))
    iterator = events.__aiter__()
    pending: asyncio.Task | None = None
    try:
        pending = asyncio.create_task(anext(iterator))
        while True:
            done, _pending = await asyncio.wait({pending}, timeout=interval)
            if not done:
                # SSE comments are transport-only. They do not enter the agent
                # event contract, trace, Context Package, or model context.
                yield ": keep-alive\n\n"
                continue
            try:
                event = pending.result()
            except StopAsyncIteration:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            pending = asyncio.create_task(anext(iterator))
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pending
        close = getattr(iterator, "aclose", None)
        if close is not None:
            with suppress(asyncio.CancelledError, Exception):
                await close()


def agent_admission_http_exception(exc: AgentAdmissionError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=exc.payload(),
        headers={"Retry-After": str(exc.retry_after_seconds)},
    )


def active_context_graph_admission_payload() -> dict:
    fixes = list(ACTIVE_CONTEXT_GRAPH_REBUILD_FIX_COMMANDS)
    return ActiveContextGraphAdmissionErrorDetail(
        protocol_version=ACTIVE_CONTEXT_GRAPH_ADMISSION_PROTOCOL_VERSION,
        code="active_context_graph_rebuild_required",
        title="Active context graph rebuild required",
        message=(
            "Retrieval is blocked because the active contextual index or one of "
            "its dependent graph layers is missing, stale, or not verified."
        ),
        reason="active_graph_freshness_gate_rejected",
        action="rebuild_context_graph",
        issues=[
            ActiveContextGraphAdmissionIssue(
                code="active_context_graph_not_admissible",
                title="Active graph freshness gate rejected retrieval",
                message=(
                    "Rebuild and verify the contextual index, relation graph, RQ "
                    "memberships, and concept graphs before retrying."
                ),
                fix_commands=fixes,
            )
        ],
        fix_commands=fixes,
        retryable=False,
        retry_after_rebuild=True,
        rebuild_required=True,
    ).model_dump()


def active_context_graph_admission_http_exception(
    exc: ActiveContextGraphAdmissionError,
) -> HTTPException:
    # The exception text can contain internal state, chunk, or point identities.
    # Preserve it only as the server-side exception cause, never in the payload.
    _ = exc
    return HTTPException(
        status_code=ACTIVE_CONTEXT_GRAPH_ADMISSION_STATUS_CODE,
        detail=active_context_graph_admission_payload(),
        headers={"Cache-Control": "no-store"},
    )


def get_requested_knowledge_base(db: Session, knowledge_base_id: str | None = None):
    try:
        return resolve_knowledge_base(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def search_model_dependency_failure_payload(exc: Exception) -> dict:
    return external_error_payload(
        exc,
        code="search_model_dependency_failed",
        title="Search model dependency failed",
        message=(
            "Query perception or query embedding failed at the configured model "
            "service. Retrieval did not use a fake or lexical-only fallback."
        ),
        fix_commands=[
            "Check the chat and embedding model settings in the repository-root .env.",
            "Verify the API container can reach both configured model services.",
        ],
    )


def gray_zone_trace_invariant_payload() -> dict:
    return {
        "code": "gray_zone_trace_invariant_failed",
        "title": "Deterministic gray-zone audit failed",
        "message": (
            "Retrieval was rejected because its deterministic gray-zone trace could not be "
            "validated before persistence. No model decision was substituted."
        ),
        "retryable": False,
        "fix_commands": [
            "Inspect the server-side gray-zone trace validation log.",
            "Rebuild or repair the active graph before retrying retrieval.",
        ],
    }


@router.post(
    "/search",
    response_model=SearchResponse,
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    from app.services.intent_execution_agent import execute_intent_search

    try:
        payload = await execute_intent_search(db, request)
        payload["results"] = [
            public_search_result_payload(item) for item in payload["results"]
        ]
        return payload
    except ConversationStateNotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ConversationStateConflictError, ConversationStateIntegrityError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ActiveContextGraphAdmissionError as exc:
        db.rollback()
        raise active_context_graph_admission_http_exception(exc) from exc
    except GrayZoneTraceInvariantError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=gray_zone_trace_invariant_payload()) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=502,
            detail=search_model_dependency_failure_payload(exc),
        ) from exc


@router.post(
    "/search/graph-enhanced",
    response_model=SearchResponse,
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def graph_search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    return await search(request, db)


@router.post(
    "/qa",
    response_model=QAResponse,
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def qa(request: QARequest) -> dict:
    from app.db import SessionLocal

    performance = QAPerformance()
    try:
        with performance.activate(), qa_stage("admission_queue"):
            admission = await acquire_agent_request_slot("qa")
        try:
            agent_request = AgentRequest(
                question=request.question,
                session_id=request.session_id,
                knowledge_base_id=request.knowledge_base_id,
                filters=request.filters,
                top_k=request.top_k,
                history=request.history,
                conversation_state_update=request.conversation_state_update,
                stream_trace=False,
            )
            with performance.activate(), SessionLocal() as db:
                return await run_agent(db, agent_request, admission=admission)
        finally:
            await admission.release()
    except AgentAdmissionError as exc:
        raise agent_admission_http_exception(exc) from exc
    except ActiveContextGraphAdmissionError as exc:
        raise active_context_graph_admission_http_exception(exc) from exc
    except (ConversationStateConflictError, ConversationStateIntegrityError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        run_id = getattr(exc, "agent_run_id", None)
        session_id = getattr(exc, "agent_session_id", None)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "agent_qa_failed",
                "message": public_exception_message(exc),
                "run_id": str(run_id) if run_id else None,
                "session_id": str(session_id) if session_id else None,
            },
        ) from exc


@router.post(
    "/qa/stream",
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def qa_stream(request: QARequest) -> StreamingResponse:
    from app.db import SessionLocal

    performance = QAPerformance()
    try:
        with performance.activate(), qa_stage("admission_queue"):
            admission = await acquire_agent_request_slot("sse")
    except AgentAdmissionError as exc:
        raise agent_admission_http_exception(exc) from exc
    try:
        with SessionLocal() as db:
            # Resolve only the knowledge-base identity before opening the SSE
            # response. Unified Agent question perception must run before the
            # retrieval-only graph admission gate so direct_answer can work
            # even when no active graph exists. Retrieval requests still fail
            # closed inside execute_agent_run and are emitted as SSE errors.
            get_requested_knowledge_base(db, request.knowledge_base_id)
        admission.raise_if_lost()
    except AgentAdmissionError as exc:
        await admission.release()
        raise agent_admission_http_exception(exc) from exc
    except ActiveContextGraphAdmissionError as exc:
        await admission.release()
        raise active_context_graph_admission_http_exception(exc) from exc
    except BaseException:
        await admission.release()
        raise
    try:
        agent_request = AgentRequest(
            question=request.question,
            session_id=request.session_id,
            knowledge_base_id=request.knowledge_base_id,
            filters=request.filters,
            top_k=request.top_k,
            history=request.history,
            conversation_state_update=request.conversation_state_update,
            stream_trace=True,
        )

        async def agent_event_objects():
            try:
                async for event in stream_agent_events(agent_request, admission=admission, performance=performance):
                    yield event
            except AgentAdmissionError as exc:
                yield {"type": "error", "error": exc.message, "detail": exc.payload()}
            except ActiveContextGraphAdmissionError:
                detail = active_context_graph_admission_payload()
                yield {"type": "error", "error": detail["message"], "detail": detail}
            except Exception as exc:
                yield {"type": "error", "error": public_exception_message(exc)}

        async def event_stream():
            async for frame in stream_sse_frames(agent_event_objects()):
                yield frame
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except BaseException:
        await admission.release()
        raise


@router.post(
    "/agent",
    response_model=AgentResponse,
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def agent_call(request: AgentRequest) -> dict:
    from app.db import SessionLocal

    performance = QAPerformance()
    try:
        with performance.activate(), qa_stage("admission_queue"):
            admission = await acquire_agent_request_slot("agent")
        try:
            with performance.activate(), SessionLocal() as db:
                return await run_agent(db, request, admission=admission)
        finally:
            await admission.release()
    except AgentAdmissionError as exc:
        raise agent_admission_http_exception(exc) from exc
    except ActiveContextGraphAdmissionError as exc:
        raise active_context_graph_admission_http_exception(exc) from exc
    except (ConversationStateConflictError, ConversationStateIntegrityError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "agent_request_failed",
                "message": public_exception_message(exc),
            },
        ) from exc


@router.get("/agent/runs/{run_id}", response_model=TaskStatusResponse)
@router.get("/tasks/{run_id}", response_model=TaskStatusResponse)
def agent_run_status(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    payload = run_to_task_status(run)
    trace_events = db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run.id).order_by(AgentTraceEvent.sequence_index.asc())).all()
    payload["trace"] = [trace_event_to_payload(event) for event in trace_events]
    return payload


@router.get(
    "/agent/runs/{run_id}/pe-audit",
    response_model=AgentPEAuditResponse,
)
def agent_run_pe_audit(
    run_id: str,
    response: Response,
    db: Session = Depends(get_db),
) -> AgentPEAuditResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return load_agent_pe_audit(db, run_id)
    except LookupError as exc:
        raise HTTPException(
            status_code=404,
            detail="Agent run not found",
            headers={"Cache-Control": "no-store"},
        ) from exc
    except AgentPEAuditIntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "agent_pe_audit_integrity_failed",
                "message": str(exc),
                "run_id": run_id,
            },
            headers={"Cache-Control": "no-store"},
        ) from exc


@router.post("/agent/runs/{run_id}/cancel", response_model=TaskStatusResponse)
@router.post("/tasks/{run_id}/cancel", response_model=TaskStatusResponse)
def agent_run_cancel(run_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        payload = cancel_agent_run(db, run_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    trace_events = db.scalars(select(AgentTraceEvent).where(AgentTraceEvent.run_id == run_id).order_by(AgentTraceEvent.sequence_index.asc())).all()
    payload["trace"] = [trace_event_to_payload(event) for event in trace_events]
    return payload

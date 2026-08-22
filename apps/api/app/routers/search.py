from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

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
    knowledge_base = get_requested_knowledge_base(db, request.knowledge_base_id)
    try:
        _session, conversation = load_conversation_state(
            db,
            knowledge_base_id=knowledge_base.id,
            session_id=request.session_id,
        )
        effective_filters = merge_search_filters_with_conversation_constraints(
            request.filters,
            conversation.active_user_constraints,
        )
        results, model_audit = await search_chunks_with_audit(
            db,
            knowledge_base.id,
            request.query,
            effective_filters,
            request.top_k,
            retrieval_granularity=request.retrieval_granularity,
            conversation_state_scope_hash=conversation.scope_hash,
            conversation_state_audit=conversation.retrieval_audit(),
            conversation_prompt_history=conversation.prompt_history,
            conversation_prompt_history_audit=(
                conversation.prompt_history_audit
            ),
        )
        public_results = [
            public_search_result_payload(item) for item in results
        ]
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
    db.commit()
    return {
        "query": request.query,
        "results": public_results,
        "degraded_mode": is_degraded_mode(),
        "model_audit": model_audit,
        "retrieval_trace_id": model_audit.get("retrieval_trace_id"),
        "context_package_id": model_audit.get("context_package_id"),
        "retrieval_granularity": request.retrieval_granularity,
        "conversation_state": conversation.public_payload(),
    }


@router.post(
    "/search/graph-enhanced",
    response_model=SearchResponse,
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def graph_search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, request.knowledge_base_id)
    try:
        _session, conversation = load_conversation_state(
            db,
            knowledge_base_id=knowledge_base.id,
            session_id=request.session_id,
        )
        effective_filters = merge_search_filters_with_conversation_constraints(
            request.filters,
            conversation.active_user_constraints,
        )
        results, audit = await layered_context_search_chunks_with_audit(
            db,
            knowledge_base.id,
            request.query,
            effective_filters,
            request.top_k,
            route="layered_context_graph",
            retrieval_granularity=request.retrieval_granularity,
            conversation_state_scope_hash=conversation.scope_hash,
            conversation_state_audit=conversation.retrieval_audit(),
            conversation_prompt_history=conversation.prompt_history,
            conversation_prompt_history_audit=(
                conversation.prompt_history_audit
            ),
        )
        public_results = [
            public_search_result_payload(item) for item in results
        ]
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
        raise HTTPException(status_code=502, detail={"code": "graph_search_failed", "message": public_exception_message(exc)}) from exc
    db.commit()
    return {
        "query": request.query,
        "results": public_results,
        "degraded_mode": is_degraded_mode(),
        "model_audit": audit,
        "retrieval_trace_id": audit.get("retrieval_trace_id"),
        "context_package_id": audit.get("context_package_id"),
        "retrieval_granularity": request.retrieval_granularity,
        "conversation_state": conversation.public_payload(),
    }


@router.post(
    "/qa",
    response_model=QAResponse,
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def qa(request: QARequest) -> dict:
    from app.db import SessionLocal

    try:
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
                retrieval_granularity=request.retrieval_granularity,
                route="layered_context_graph",
                stream_trace=False,
            )
            with SessionLocal() as db:
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
        raise HTTPException(
            status_code=502,
            detail={
                "code": "agent_qa_failed",
                "message": public_exception_message(exc),
            },
        ) from exc


@router.post(
    "/qa/stream",
    responses=ACTIVE_CONTEXT_GRAPH_ADMISSION_RESPONSES,
)
async def qa_stream(request: QARequest) -> StreamingResponse:
    from app.db import SessionLocal

    try:
        admission = await acquire_agent_request_slot("sse")
    except AgentAdmissionError as exc:
        raise agent_admission_http_exception(exc) from exc
    try:
        with SessionLocal() as db:
            knowledge_base = get_requested_knowledge_base(db, request.knowledge_base_id)
            try:
                active_graph_admission_gate(db, knowledge_base.id)
            except ActiveContextGraphAdmissionError as exc:
                db.rollback()
                raise
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
            retrieval_granularity=request.retrieval_granularity,
            route="layered_context_graph",
            stream_trace=True,
        )

        async def event_stream():
            try:
                async for event in stream_agent_events(agent_request, admission=admission):
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            except AgentAdmissionError as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': exc.message, 'detail': exc.payload()}, ensure_ascii=False)}\n\n"
            except ActiveContextGraphAdmissionError:
                detail = active_context_graph_admission_payload()
                yield f"data: {json.dumps({'type': 'error', 'error': detail['message'], 'detail': detail}, ensure_ascii=False)}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'error': public_exception_message(exc)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream", background=BackgroundTask(admission.release))
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

    try:
        admission = await acquire_agent_request_slot("agent")
        try:
            with SessionLocal() as db:
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

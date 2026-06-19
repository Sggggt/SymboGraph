from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AgentRun
from app.schemas import AgentRequest, AgentResponse, QARequest, QAResponse, SearchRequest, SearchResponse, TaskStatusResponse
from app.services.agent_graph import run_agent, run_to_task_status, stream_agent_events
from app.services.embeddings import is_degraded_mode
from app.services.error_sanitizer import external_error_payload, public_exception_message
from app.services.ingestion import resolve_knowledge_base
from app.services.retrieval import layered_context_search_chunks_with_audit, search_chunks_with_audit

router = APIRouter()


def get_requested_knowledge_base(db: Session, knowledge_base_id: str | None = None):
    try:
        return resolve_knowledge_base(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def embedding_failure_payload(exc: Exception) -> dict:
    return external_error_payload(
        exc,
        code="search_embedding_failed",
        title="Search embedding request failed",
        message="The query could not be embedded by the configured model API. Retrieval did not fall back to fake or lexical-only results.",
        fix_commands=[
            "Check EMBEDDING_BASE_URL and EMBEDDING_RESOLVE_IP in .env.",
            "Verify the API container can reach the embedding endpoint.",
        ],
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, request.knowledge_base_id)
    try:
        results, model_audit = await search_chunks_with_audit(db, knowledge_base.id, request.query, request.filters, request.top_k)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail=embedding_failure_payload(exc)) from exc
    db.commit()
    return {"query": request.query, "results": results, "degraded_mode": is_degraded_mode(), "model_audit": model_audit}


@router.post("/search/graph-enhanced", response_model=SearchResponse)
async def graph_search(request: SearchRequest, db: Session = Depends(get_db)) -> dict:
    knowledge_base = get_requested_knowledge_base(db, request.knowledge_base_id)
    try:
        results, audit = await layered_context_search_chunks_with_audit(
            db,
            knowledge_base.id,
            request.query,
            request.filters,
            request.top_k,
            route="layered_context_graph",
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=502, detail={"code": "graph_search_failed", "message": public_exception_message(exc)}) from exc
    db.commit()
    return {"query": request.query, "results": results, "degraded_mode": is_degraded_mode(), "model_audit": audit}


@router.post("/qa", response_model=QAResponse)
async def qa(request: QARequest, db: Session = Depends(get_db)) -> dict:
    get_requested_knowledge_base(db, request.knowledge_base_id)
    return await run_agent(
        db,
        AgentRequest(
            question=request.question,
            session_id=request.session_id,
            knowledge_base_id=request.knowledge_base_id,
            filters=request.filters,
            top_k=request.top_k,
            history=request.history,
            route="layered_context_graph",
            stream_trace=False,
        ),
    )


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
        route="layered_context_graph",
        stream_trace=True,
    )

    async def event_stream():
        db = SessionLocal()
        try:
            async for event in stream_agent_events(db, agent_request):
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': public_exception_message(exc)}, ensure_ascii=False)}\n\n"
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

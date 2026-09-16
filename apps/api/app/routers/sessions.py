from __future__ import annotations

from collections import Counter
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AgentRun, AnswerSession, QASession
from app.schemas import DeleteResponse, SessionMessagesResponse, SessionSummary
from app.services.ingestion import resolve_knowledge_base
from app.services.conversation_state import (
    ConversationStateIntegrityError,
    load_conversation_state,
    session_transcript_public_payload,
    session_summary_payload,
)

router = APIRouter()
logger = logging.getLogger(__name__)
INCOMPATIBLE_SESSION_DETAIL = (
    "Session history is not compatible with the current public protocol"
)


def get_requested_knowledge_base(db: Session, knowledge_base_id: str | None = None):
    try:
        return resolve_knowledge_base(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> list[dict]:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    sessions = [
        session
        for session in db.scalars(
            select(QASession)
            .where(QASession.knowledge_base_id == knowledge_base.id)
            .order_by(QASession.updated_at.desc())
        ).all()
        if session.transcript
    ]
    compatible: list[dict] = []
    excluded_by_error: Counter[str] = Counter()
    for session in sessions:
        try:
            payload = session_summary_payload(
                db,
                session,
                validate_references=False,
            )
            SessionSummary.model_validate(payload)
            compatible.append(payload)
        except (ConversationStateIntegrityError, ValidationError) as exc:
            # One legacy or corrupt session is Byzantine input to this list,
            # not authority to make every compatible session unavailable.
            excluded_by_error[type(exc).__name__] += 1
    if excluded_by_error:
        logger.warning(
            "Excluded incompatible sessions from public session list",
            extra={
                "excluded_session_count": sum(excluded_by_error.values()),
                "excluded_error_types": dict(sorted(excluded_by_error.items())),
            },
        )
    return compatible


@router.get("/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.get(QASession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        payload = session_summary_payload(db, session, validate_references=False)
        SessionSummary.model_validate(payload)
        return payload
    except (ConversationStateIntegrityError, ValidationError) as exc:
        raise HTTPException(
            status_code=409,
            detail=INCOMPATIBLE_SESSION_DETAIL,
        ) from exc


@router.get("/sessions/{session_id}/messages", response_model=SessionMessagesResponse)
def get_session_messages(session_id: str, db: Session = Depends(get_db)) -> dict:
    session = db.get(QASession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        _session, conversation = load_conversation_state(
            db,
            knowledge_base_id=session.knowledge_base_id,
            session_id=session.id,
            validate_references=False,
        )
    except ConversationStateIntegrityError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    payload = {
        "session_id": session.id,
        "messages": session_transcript_public_payload(db, session),
        "conversation_state": conversation.public_payload(),
    }
    try:
        SessionMessagesResponse.model_validate(payload)
        return payload
    except ValidationError as exc:
        raise HTTPException(
            status_code=409,
            detail=INCOMPATIBLE_SESSION_DETAIL,
        ) from exc


@router.delete("/sessions/{session_id}", response_model=DeleteResponse)
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    # The transcript is user-owned session state. Runs, answer sessions,
    # source bindings and observations are durable audit facts whose foreign
    # keys detach with ON DELETE SET NULL. PostgreSQL therefore needs one
    # DELETE ... RETURNING round trip. SQLite test/fallback engines may run
    # without FK actions, so they receive the equivalent explicit detachment.
    if db.get_bind().dialect.name == "sqlite":
        session = db.get(QASession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        db.query(AgentRun).filter(AgentRun.session_id == session_id).update(
            {AgentRun.session_id: None},
            synchronize_session="fetch",
        )
        db.query(AnswerSession).filter(
            AnswerSession.qa_session_id == session_id
        ).update(
            {AnswerSession.qa_session_id: None},
            synchronize_session="fetch",
        )
        db.delete(session)
    else:
        deleted_id = db.scalar(
            delete(QASession)
            .where(QASession.id == session_id)
            .returning(QASession.id)
        )
        if deleted_id is None:
            raise HTTPException(status_code=404, detail="Session not found")
    db.commit()
    return {"deleted": True}

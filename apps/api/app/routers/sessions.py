from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import AgentRun, AgentTraceEvent, QASession
from app.schemas import DeleteResponse, SessionMessagesResponse, SessionSummary
from app.services.ingestion import resolve_knowledge_base

router = APIRouter()


def get_requested_knowledge_base(db: Session, knowledge_base_id: str | None = None):
    try:
        return resolve_knowledge_base(db, knowledge_base_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(knowledge_base_id: str | None = None, db: Session = Depends(get_db)) -> list[QASession]:
    knowledge_base = get_requested_knowledge_base(db, knowledge_base_id)
    return list(db.scalars(select(QASession).where(QASession.knowledge_base_id == knowledge_base.id).order_by(QASession.updated_at.desc())).all())


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

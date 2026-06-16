from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import (
    DeleteResponse,
    KnowledgeBaseSummary,
    ModelSettingsResponse,
    ModelSettingsUpdate,
    RuntimeCheckResponse,
    StrategyProfileAssistantRequest,
    StrategyProfileAssistantStateResponse,
    StrategyProfileBindRequest,
    StrategyProfileCopyRequest,
    StrategyProfileCreateRequest,
    StrategyProfileDetail,
    StrategyProfileMutationResponse,
    StrategyProfileSummary,
    StrategyProfileUpdateRequest,
)
from app.services.ingestion import summarize_knowledge_base
from app.services.profile_assistant import get_profile_assistant_state, stream_profile_assistant_events
from app.services.runtime_settings import model_settings_payload, runtime_check_payload, update_model_settings
from app.services.strategy_profiles import (
    bind_profile_to_knowledge_base,
    copy_profile,
    create_profile,
    delete_profile,
    get_profile_or_raise,
    list_profiles,
    profile_to_payload,
    update_profile,
)

router = APIRouter()


@router.get("/settings/model", response_model=ModelSettingsResponse)
def get_model_settings() -> dict:
    return model_settings_payload()


@router.put("/settings/model", response_model=ModelSettingsResponse)
def save_model_settings(request: ModelSettingsUpdate) -> dict:
    try:
        return update_model_settings(request.model_dump(exclude_unset=True))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
        knowledge_base = bind_profile_to_knowledge_base(db, knowledge_base_id=request.knowledge_base_id, profile_id=request.profile_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return summarize_knowledge_base(db, knowledge_base)


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

    return StreamingResponse(event_stream(), media_type="text/event-stream")

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
    RuntimeSettingsCandidateActionRequest,
    RuntimeSettingsCandidateCreate,
    RuntimeSettingsCandidateResponse,
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
from app.services.runtime_settings import (
    model_settings_payload,
    runtime_check_payload,
    save_model_settings_to_root_env,
)
from app.services.runtime_settings_lifecycle import (
    apply_runtime_settings_activation_intent,
    build_runtime_settings_shadow,
    evaluate_runtime_settings_shadow,
    preview_runtime_settings_candidate,
    promote_runtime_settings_candidate,
    record_runtime_settings_build_failure,
    rollback_runtime_settings_candidate,
    runtime_settings_candidate_payload,
    stage_runtime_settings_candidate,
)
from app.services.strategy_profiles import (
    bind_profile_to_knowledge_base,
    copy_profile,
    create_profile,
    delete_profile,
    get_profile_or_raise,
    list_profiles,
    ProfileIntegrityError,
    profile_to_payload,
    update_profile,
)

router = APIRouter()


@router.get("/settings/model", response_model=ModelSettingsResponse)
def get_model_settings() -> dict:
    return model_settings_payload()


@router.put("/settings/model", response_model=ModelSettingsResponse)
def save_model_settings(
    request: ModelSettingsUpdate,
    db: Session = Depends(get_db),
) -> dict:
    try:
        return save_model_settings_to_root_env(
            db,
            request.model_dump(exclude_unset=True),
        )
    except (ValueError, RuntimeError) as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/settings/runtime-check", response_model=RuntimeCheckResponse)
def get_runtime_check() -> dict:
    return runtime_check_payload()


@router.post(
    "/settings/runtime-candidates",
    response_model=RuntimeSettingsCandidateResponse,
)
def create_runtime_settings_candidate(
    request: RuntimeSettingsCandidateCreate,
    db: Session = Depends(get_db),
) -> dict:
    try:
        preview = preview_runtime_settings_candidate(
            db,
            knowledge_base_ids=request.knowledge_base_ids,
            requested_settings=request.settings,
        )
        if request.dry_run_only:
            db.rollback()
            return {"preview": preview, "action": {"staged": False, "dry_run_only": True}}
        candidate, _builds = stage_runtime_settings_candidate(
            db,
            knowledge_base_ids=request.knowledge_base_ids,
            requested_settings=request.settings,
            source=request.source,
        )
        db.commit()
        return {
            "candidate": runtime_settings_candidate_payload(db, candidate.id),
            "preview": preview,
            "action": {"staged": True, "active_mutated": False},
        }
    except (ValueError, RuntimeError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/settings/runtime-candidates/{candidate_id}",
    response_model=RuntimeSettingsCandidateResponse,
)
def get_runtime_settings_candidate(
    candidate_id: str,
    db: Session = Depends(get_db),
) -> dict:
    try:
        return {"candidate": runtime_settings_candidate_payload(db, candidate_id)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/settings/runtime-candidates/{candidate_id}/build",
    response_model=RuntimeSettingsCandidateResponse,
)
async def build_runtime_settings_candidate(
    candidate_id: str,
    request: RuntimeSettingsCandidateActionRequest,
    db: Session = Depends(get_db),
) -> dict:
    try:
        payload = runtime_settings_candidate_payload(db, candidate_id)
        build_ids = [
            str(item["id"])
            for item in payload.get("builds") or []
            if request.build_id is None or str(item["id"]) == request.build_id
        ]
        if not build_ids:
            raise ValueError("No matching Runtime Settings shadow build")
        from app.core.config import get_settings

        if get_settings().ingestion_execution_mode != "inline":
            from app.models import RuntimeSettingsShadowBuild
            from sqlalchemy import select

            rows = list(
                db.scalars(
                    select(RuntimeSettingsShadowBuild)
                    .where(RuntimeSettingsShadowBuild.id.in_(build_ids))
                    .with_for_update()
                ).all()
            )
            if len(rows) != len(build_ids):
                raise RuntimeError("Runtime Settings build scope changed before enqueue")
            for row in rows:
                row.diagnostics_json = {
                    **dict(row.diagnostics_json or {}),
                    "worker_enqueue_status": "pending",
                }
            db.commit()
            try:
                from worker_app.tasks import build_runtime_settings_candidate_task

                task = build_runtime_settings_candidate_task.apply_async(
                    args=[candidate_id, build_ids],
                    queue=get_settings().ingestion_task_queue,
                )
            except Exception as exc:
                for row in rows:
                    db.refresh(row)
                    row.diagnostics_json = {
                        **dict(row.diagnostics_json or {}),
                        "worker_enqueue_status": "failed",
                        "worker_enqueue_error_type": type(exc).__name__,
                    }
                db.commit()
                raise RuntimeError(
                    "Runtime Settings shadow build enqueue failed: "
                    f"error_type={type(exc).__name__}"
                ) from None
            task_id = str(getattr(task, "id", "") or "")
            for row in rows:
                db.refresh(row)
                row.diagnostics_json = {
                    **dict(row.diagnostics_json or {}),
                    "worker_enqueue_status": "enqueued",
                    "worker_task_id": task_id,
                }
            db.commit()
            return {
                "candidate": runtime_settings_candidate_payload(db, candidate_id),
                "action": {
                    "enqueued": True,
                    "task_id": task_id,
                    "build_ids": build_ids,
                    "active_mutated": False,
                },
            }
        for build_id in build_ids:
            try:
                await build_runtime_settings_shadow(db, build_id=build_id)
                db.commit()
            except Exception as exc:
                db.rollback()
                record_runtime_settings_build_failure(
                    db,
                    build_id=build_id,
                    error_type=type(exc).__name__,
                )
                db.commit()
                raise RuntimeError(
                    f"Runtime Settings shadow build failed: build_id={build_id}; "
                    f"error_type={type(exc).__name__}"
                ) from None
        return {
            "candidate": runtime_settings_candidate_payload(db, candidate_id),
            "action": {"built": True, "build_ids": build_ids, "active_mutated": False},
        }
    except (ValueError, RuntimeError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/settings/runtime-candidates/{candidate_id}/evaluate",
    response_model=RuntimeSettingsCandidateResponse,
)
def evaluate_runtime_settings_candidate(
    candidate_id: str,
    request: RuntimeSettingsCandidateActionRequest,
    db: Session = Depends(get_db),
) -> dict:
    try:
        payload = runtime_settings_candidate_payload(db, candidate_id)
        build_ids = [
            str(item["id"])
            for item in payload.get("builds") or []
            if request.build_id is None or str(item["id"]) == request.build_id
        ]
        if not build_ids:
            raise ValueError("No matching Runtime Settings shadow build")
        for build_id in build_ids:
            evaluate_runtime_settings_shadow(db, build_id=build_id)
        db.commit()
        return {
            "candidate": runtime_settings_candidate_payload(db, candidate_id),
            "action": {"evaluated": True, "build_ids": build_ids, "active_mutated": False},
        }
    except (ValueError, RuntimeError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/settings/runtime-candidates/{candidate_id}/promote",
    response_model=RuntimeSettingsCandidateResponse,
)
def promote_runtime_settings_candidate_route(
    candidate_id: str,
    db: Session = Depends(get_db),
) -> dict:
    try:
        promotion = promote_runtime_settings_candidate(db, candidate_id)
        db.commit()
        activation = None
        if promotion.get("promoted") and promotion.get("activation_intent_id"):
            activation = apply_runtime_settings_activation_intent(
                str(promotion["activation_intent_id"])
            )
            db.expire_all()
        return {
            "candidate": runtime_settings_candidate_payload(db, candidate_id),
            "action": {"promotion": promotion, "activation": activation},
        }
    except (ValueError, RuntimeError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/settings/runtime-candidates/{candidate_id}/rollback",
    response_model=RuntimeSettingsCandidateResponse,
)
def rollback_runtime_settings_candidate_route(
    candidate_id: str,
    request: RuntimeSettingsCandidateActionRequest,
    db: Session = Depends(get_db),
) -> dict:
    try:
        candidate = rollback_runtime_settings_candidate(
            db,
            candidate_id,
            reason=str(request.reason or ""),
        )
        db.commit()
        payload = runtime_settings_candidate_payload(db, candidate.id)
        intents = [
            item
            for item in payload.get("activation_intents") or []
            if item.get("direction") == "rollback"
        ]
        if not intents:
            raise RuntimeError("Rollback did not create its activation intent")
        activation = apply_runtime_settings_activation_intent(str(intents[-1]["id"]))
        db.expire_all()
        return {
            "candidate": runtime_settings_candidate_payload(db, candidate.id),
            "action": {"rolled_back": True, "activation": activation},
        }
    except (ValueError, RuntimeError) as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/settings/runtime-activation-intents/{intent_id}/apply",
    response_model=RuntimeSettingsCandidateResponse,
)
def apply_runtime_settings_intent_route(intent_id: str) -> dict:
    try:
        return {"action": {"activation": apply_runtime_settings_activation_intent(intent_id)}}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/settings/profiles", response_model=list[StrategyProfileSummary])
def get_strategy_profiles(db: Session = Depends(get_db)) -> list[dict]:
    return list_profiles(db)


@router.get("/settings/profiles/{profile_id}", response_model=StrategyProfileDetail)
def get_strategy_profile(profile_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        return profile_to_payload(get_profile_or_raise(db, profile_id))
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProfileIntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/settings/profiles", response_model=StrategyProfileMutationResponse)
def create_strategy_profile(request: StrategyProfileCreateRequest, db: Session = Depends(get_db)) -> dict:
    try:
        profile, warnings = create_profile(
            db,
            name=request.name,
            library_type=request.library_type,
            profile_json=request.profile_json,
        )
        public_profile = profile_to_payload(profile)
    except ProfileIntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"profile": public_profile, "warnings": warnings}


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
        public_profile = profile_to_payload(profile)
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProfileIntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"profile": public_profile, "warnings": warnings}


@router.post("/settings/profiles/{profile_id}/copy", response_model=StrategyProfileDetail)
def copy_strategy_profile(profile_id: str, request: StrategyProfileCopyRequest, db: Session = Depends(get_db)) -> dict:
    try:
        return profile_to_payload(copy_profile(db, profile_id, name=request.name))
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProfileIntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/settings/profiles/{profile_id}", response_model=DeleteResponse)
def delete_strategy_profile(profile_id: str, db: Session = Depends(get_db)) -> dict:
    try:
        delete_profile(db, profile_id)
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProfileIntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"deleted": True}


@router.post("/settings/profiles/bind", response_model=KnowledgeBaseSummary)
def bind_strategy_profile(request: StrategyProfileBindRequest, db: Session = Depends(get_db)) -> dict:
    try:
        knowledge_base = bind_profile_to_knowledge_base(db, knowledge_base_id=request.knowledge_base_id, profile_id=request.profile_id)
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ProfileIntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
            db.rollback()
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ProfileIntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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

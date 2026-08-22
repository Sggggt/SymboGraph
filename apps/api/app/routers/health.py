from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.services.embeddings import is_degraded_mode
from app.services.storage import storage_deployment_contract
from app.services.storage_maintenance import storage_maintenance_recovery_health
from app.services.upload_replacement import upload_replacement_recovery_health

router = APIRouter()


@router.get("/ready")
def readiness(request: Request) -> dict:
    if getattr(request.app.state, "startup_ready", False) is not True:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "startup_not_ready",
                "message": "API startup lifecycle has not completed",
            },
        )
    return {
        "status": "ready",
        "runtime": "four_layer_context_graph",
    }


@router.get("/health")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "degraded_mode": is_degraded_mode(),
        "runtime": "four_layer_context_graph",
        "storage_deployment": storage_deployment_contract(),
        "storage_maintenance_recovery": storage_maintenance_recovery_health(),
        "upload_replacement_recovery": upload_replacement_recovery_health(),
    }

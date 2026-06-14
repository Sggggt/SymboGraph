from __future__ import annotations

from fastapi import APIRouter

from app.services.embeddings import is_degraded_mode

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict:
    return {"status": "ok", "degraded_mode": is_degraded_mode(), "runtime": "four_layer_context_graph"}

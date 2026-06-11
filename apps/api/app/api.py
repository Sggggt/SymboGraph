from __future__ import annotations

from fastapi import APIRouter

from app.routers import health, ingestion, knowledge, search, sessions, settings
from app.routers.knowledge import rebuild_graph_endpoint

router = APIRouter()
for subrouter in (
    health.router,
    settings.router,
    knowledge.router,
    ingestion.router,
    search.router,
    sessions.router,
):
    router.include_router(subrouter)

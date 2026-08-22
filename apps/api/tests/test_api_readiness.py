from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_readiness_fails_closed_until_lifespan_marks_startup_complete() -> None:
    from app.routers.health import router

    app = FastAPI()
    app.state.startup_ready = False
    app.include_router(router, prefix="/api")

    with TestClient(app) as client:
        response = client.get("/api/ready")
        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "code": "startup_not_ready",
                "message": "API startup lifecycle has not completed",
            }
        }

        app.state.startup_ready = True
        response = client.get("/api/ready")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "runtime": "four_layer_context_graph",
        }


def test_compose_api_uses_readiness_with_source_mounted_runtime_reload() -> None:
    compose = (
        Path(__file__).resolve().parents[3] / "infra" / "docker-compose.yml"
    ).read_text(encoding="utf-8")
    api_section = compose.split("\n  api:\n", 1)[1].split("\n  worker:\n", 1)[0]

    assert '"http://127.0.0.1:8000/api/ready"' in api_section
    assert '"--reload"' in api_section
    assert '"--reload-dir"' in api_section
    assert '"/app/apps/api/app"' in api_section
    assert "../apps/api/app:/app/apps/api/app:ro" in api_section

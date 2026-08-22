from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


class _TransactionSpy:
    def __init__(self) -> None:
        self.rollback_calls = 0
        self.commit_calls = 0

    def rollback(self) -> None:
        self.rollback_calls += 1

    def commit(self) -> None:
        self.commit_calls += 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route_name", "service_name"),
    [
        ("search", "search_chunks_with_audit"),
        ("graph_search", "layered_context_search_chunks_with_audit"),
    ],
)
async def test_search_routes_return_typed_sanitized_graph_admission_conflict(
    monkeypatch: pytest.MonkeyPatch,
    route_name: str,
    service_name: str,
) -> None:
    from app.routers import search as search_router
    from app.schemas import (
        ActiveContextGraphAdmissionErrorDetail,
        SearchRequest,
    )
    from app.services.context_graph import ActiveContextGraphAdmissionError

    secret = "Authorization: Bearer sk-secret"
    internal_chunk_id = "chunk-internal-123"
    knowledge_base_id = "11111111-1111-4111-8111-111111111111"

    async def reject_stale_graph(*args, **kwargs):
        raise ActiveContextGraphAdmissionError(
            f"{secret}; active_vector_records_missing:{internal_chunk_id}"
        )

    monkeypatch.setattr(
        search_router,
        "get_requested_knowledge_base",
        lambda db, knowledge_base_id=None: SimpleNamespace(id=knowledge_base_id),
    )
    monkeypatch.setattr(search_router, service_name, reject_stale_graph)
    db = _TransactionSpy()

    with pytest.raises(HTTPException) as captured:
        await getattr(search_router, route_name)(
            SearchRequest(
                query="should fail before retrieval",
                knowledge_base_id=knowledge_base_id,
            ),
            db,
        )

    exc = captured.value
    assert exc.status_code == 409
    assert exc.headers == {"Cache-Control": "no-store"}
    payload = ActiveContextGraphAdmissionErrorDetail.model_validate(exc.detail)
    assert payload.protocol_version == "active_context_graph_admission_error_v1"
    assert payload.code == "active_context_graph_rebuild_required"
    assert payload.reason == "active_graph_freshness_gate_rejected"
    assert payload.action == "rebuild_context_graph"
    assert payload.retryable is False
    assert payload.retry_after_rebuild is True
    assert payload.rebuild_required is True
    assert payload.issues[0].code == "active_context_graph_not_admissible"
    serialized = json.dumps(exc.detail, ensure_ascii=False)
    assert secret not in serialized
    assert "sk-secret" not in serialized
    assert internal_chunk_id not in serialized
    assert "search_embedding_failed" not in serialized
    assert "search_model_dependency_failed" not in serialized
    assert "graph_search_failed" not in serialized
    assert db.rollback_calls == 1
    assert db.commit_calls == 0


def test_graph_admission_payload_is_stable_and_contains_no_dynamic_exception_fields() -> None:
    from app.routers.search import active_context_graph_admission_payload
    from app.schemas import ActiveContextGraphAdmissionErrorDetail

    payload = active_context_graph_admission_payload()
    validated = ActiveContextGraphAdmissionErrorDetail.model_validate(payload)

    assert validated.model_dump() == payload
    assert set(payload) == {
        "protocol_version",
        "code",
        "title",
        "message",
        "reason",
        "action",
        "issues",
        "fix_commands",
        "retryable",
        "retry_after_rebuild",
        "rebuild_required",
    }

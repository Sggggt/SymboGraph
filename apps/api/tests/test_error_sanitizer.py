from __future__ import annotations

import json
import types

import httpx


def test_provider_error_summary_does_not_expose_raw_response_body():
    from app.services.embeddings import _external_provider_error_from_response, _is_unsupported_parameter_error
    from app.services.error_sanitizer import public_exception_message

    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://provider.example/v1/chat/completions"),
        json={
            "error": {
                "message": "Authorization: Bearer sk-provider-secret failed because dimensions is unsupported",
                "code": "invalid_request_error",
                "param": "dimensions",
            }
        },
    )

    exc = _external_provider_error_from_response(response, phase="unit_test")
    message = public_exception_message(exc)

    assert "sk-provider-secret" not in message
    assert "Authorization" not in message
    assert "dimensions is unsupported" not in message
    assert "http_status=400" in message
    assert "error_code=invalid_request_error" in message
    assert _is_unsupported_parameter_error(exc, "dimensions") is True


def test_search_error_payload_uses_sanitized_external_error():
    from app.routers.search import embedding_failure_payload
    from app.services.error_sanitizer import ExternalServiceError

    exc = ExternalServiceError(
        service="model_provider",
        phase="http_json",
        status_code=401,
        error_code='invalid_api_key" api_key="sk-provider-secret',
        retryable=False,
    )

    payload = embedding_failure_payload(exc)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "sk-provider-secret" not in serialized
    assert "response" not in serialized.lower()
    assert "http_status=401" in serialized


def test_model_bridge_reload_error_does_not_return_raw_body(monkeypatch):
    from app.services import runtime_settings

    settings = types.SimpleNamespace(
        model_bridge_enabled=True,
        chat_base_url="http://127.0.0.1:18000",
        model_bridge_port=18000,
        model_bridge_admin_token="unit-token",
        model_request_timeout_seconds=30,
    )

    def fake_post(*args, **kwargs):
        return types.SimpleNamespace(status_code=500, text='{"error":"Authorization: Bearer sk-provider-secret"}')

    monkeypatch.setattr(runtime_settings, "_desired_bridge_config", lambda settings, env_entries: {"chat_target_base_url": "https://chat.example/v1", "embedding_target_base_url": "https://embedding.example/v1"})
    monkeypatch.setattr(runtime_settings, "_bridge_self_target_keys", lambda desired, settings: [])
    monkeypatch.setattr(runtime_settings.httpx, "post", fake_post)

    payload = runtime_settings.reload_model_bridge(settings=settings, env_entries={})

    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["ok"] is False
    assert payload["status_code"] == 500
    assert "sk-provider-secret" not in serialized
    assert "Authorization" not in serialized

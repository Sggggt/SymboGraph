from __future__ import annotations

import json
import types

import httpx
import pytest


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
    from app.routers.search import search_model_dependency_failure_payload
    from app.services.error_sanitizer import ExternalServiceError

    exc = ExternalServiceError(
        service="model_provider",
        phase="http_json",
        status_code=401,
        error_code='invalid_api_key" api_key="sk-provider-secret',
        retryable=False,
    )

    payload = search_model_dependency_failure_payload(exc)
    serialized = json.dumps(payload, ensure_ascii=False)

    assert "sk-provider-secret" not in serialized
    assert "response" not in serialized.lower()
    assert "http_status=401" in serialized
    assert payload["code"] == "search_model_dependency_failed"


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

    monkeypatch.setattr(runtime_settings, "_desired_bridge_config", lambda settings, env_entries: {"chat_api_protocol": "openai", "embedding_api_protocol": "openai", "chat_target_base_url": "https://chat.example/v1", "chat_resolve_ip": "", "embedding_target_base_url": "https://embedding.example/v1", "embedding_resolve_ip": "", "timeout": 30})
    monkeypatch.setattr(runtime_settings, "_bridge_self_target_keys", lambda desired, settings: [])
    monkeypatch.setattr(runtime_settings, "_bridge_http_request", fake_post)

    payload = runtime_settings.reload_model_bridge(settings=settings, env_entries={})

    serialized = json.dumps(payload, ensure_ascii=False)
    assert payload["ok"] is False
    assert payload["status_code"] == 500
    assert "sk-provider-secret" not in serialized
    assert "Authorization" not in serialized


def test_model_bridge_admin_request_fails_closed_without_configured_token():
    from app.services import runtime_settings

    settings = types.SimpleNamespace(model_bridge_admin_token="")

    with pytest.raises(RuntimeError, match="MODEL_BRIDGE_ADMIN_TOKEN"):
        runtime_settings._bridge_admin_headers(settings)


def test_model_bridge_status_does_not_publish_raw_transport_exception(monkeypatch):
    from app.services import runtime_settings

    sentinel = "Authorization: Bearer status-secret provider-body"
    settings = types.SimpleNamespace(
        model_bridge_enabled=True,
        chat_base_url="http://127.0.0.1:18000",
        model_bridge_port=18000,
        model_bridge_admin_token="unit-token",
        model_request_timeout_seconds=30,
    )
    env_entries = {
        "CHAT_API_PROTOCOL": "openai",
        "CHAT_BASE_URL": "https://chat.example.test/v1",
        "CHAT_RESOLVE_IP": "1.1.1.1",
        "EMBEDDING_API_PROTOCOL": "openai",
        "EMBEDDING_BASE_URL": "https://embedding.example.test/v1",
        "EMBEDDING_RESOLVE_IP": "2.2.2.2",
    }
    monkeypatch.setattr(
        runtime_settings,
        "_bridge_http_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(sentinel)),
    )

    payload = runtime_settings.model_bridge_status_payload(
        settings=settings,
        env_entries=env_entries,
    )

    serialized = json.dumps(payload, ensure_ascii=False)
    assert sentinel not in serialized
    assert "status-secret" not in serialized
    assert payload["reachable"] is False
    assert payload["warnings"] == [
        "Model bridge health check failed with a transport error."
    ]


def test_public_exception_message_keeps_transport_error_type_when_message_empty():
    from app.services.error_sanitizer import public_exception_message

    exc = httpx.ConnectError("", request=httpx.Request("POST", "https://provider.example/v1/chat/completions"))

    assert public_exception_message(exc) == "ConnectError"


def test_external_failure_classification_replays_nested_scalar_evidence_only():
    from app.services.error_sanitizer import (
        ExternalServiceError,
        external_failure_classification,
    )

    provider = ExternalServiceError(
        service="model_provider",
        phase="pinned_https_json",
        status_code=405,
        error_code="method_not_allowed",
        retryable=False,
    )
    try:
        try:
            raise provider
        except ExternalServiceError as exc:
            raise RuntimeError(
                "outer wrapper with Authorization: Bearer secret and provider body"
            ) from exc
    except RuntimeError as wrapped:
        card = external_failure_classification(wrapped)

    assert card == {
        "protocol_version": "external_failure_classification_v1",
        "classified": True,
        "classification_source": "external_service_error",
        "outer_error_type": "RuntimeError",
        "classified_error_type": "ExternalServiceError",
        "cause_depth": 1,
        "service": "model_provider",
        "phase": "pinned_https_json",
        "http_status": 405,
        "error_code": "method_not_allowed",
        "retryable": False,
    }
    serialized = json.dumps(card, sort_keys=True)
    assert "secret" not in serialized
    assert "provider body" not in serialized
    assert "authorization" not in serialized.lower()


def test_external_failure_classification_rejects_noncanonical_error_code():
    from app.services.error_sanitizer import (
        ExternalServiceError,
        external_failure_classification,
    )

    card = external_failure_classification(
        ExternalServiceError(
            service="model_provider",
            phase="http_json",
            status_code=401,
            error_code='invalid_api_key api_key="secret-value"',
            retryable=False,
        )
    )

    assert card["error_code"] == "noncanonical_error_code"
    assert "secret-value" not in json.dumps(card, sort_keys=True)

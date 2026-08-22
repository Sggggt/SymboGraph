from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "probe_embedding_provider.py"


def _load_script():
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "probe_embedding_provider_under_test",
            SCRIPT_PATH,
        )
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT_PATH.parent))


def _args(*, execute: bool) -> Namespace:
    return Namespace(
        execute=execute,
        arm="provider",
        text="private probe text",
        text_type="query",
        timeout_seconds=10.0,
    )


def test_probe_input_card_is_content_free_and_bounded():
    module = _load_script()

    card = module.probe_input_card("private probe text", text_type="query")

    assert card["text_type"] == "query"
    assert card["text_persisted"] is False
    assert len(card["sha256"]) == 64
    assert "private probe text" not in json.dumps(card)
    with pytest.raises(ValueError, match="must not be empty"):
        module.probe_input_card("  ", text_type="query")
    with pytest.raises(ValueError, match="control character"):
        module.probe_input_card("bad\x00text", text_type="query")


@pytest.mark.asyncio
async def test_probe_dry_run_makes_no_bridge_or_provider_call(monkeypatch):
    module = _load_script()
    monkeypatch.setattr(
        module,
        "current_runtime_card",
        lambda: {
            "embedding_api_protocol": "openai",
            "embedding_model": "fixture-model",
            "embedding_dimensions": 3,
            "embedding_base_url_present": True,
            "embedding_api_key_present": True,
            "model_bridge_enabled": True,
            "model_fallback_enabled": False,
        },
    )

    async def forbidden_request(*_args, **_kwargs):
        raise AssertionError("dry-run constructed a provider request")

    monkeypatch.setattr(module, "execute_embedding_probe", forbidden_request)
    monkeypatch.setattr(
        module,
        "sync_model_runtime_card",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run synced bridge")),
    )

    payload, exit_code = await module.run_probe(_args(execute=False))

    assert exit_code == 0
    assert payload["status"] == "planned"
    assert payload["network_call_count"] == 0
    assert payload["input"]["text_persisted"] is False


@pytest.mark.asyncio
async def test_probe_execute_reports_vector_shape_without_values(monkeypatch):
    module = _load_script()
    monkeypatch.setattr(
        module,
        "current_runtime_card",
        lambda: {
            "embedding_api_protocol": "openai",
            "embedding_model": "fixture-model",
            "embedding_dimensions": 3,
            "embedding_base_url_present": True,
            "embedding_api_key_present": True,
            "model_bridge_enabled": True,
            "model_fallback_enabled": False,
        },
    )
    monkeypatch.setattr(
        module,
        "sync_model_runtime_card",
        lambda: {"attempted": True, "ok": True, "embedding_target_hash": "a" * 64},
    )

    async def successful_request(*_args, **_kwargs):
        return SimpleNamespace(
            vectors=[[1.0, 0.0, 0.0]],
            provider="openai_compatible",
            external_called=True,
            fallback_reason=None,
        )

    monkeypatch.setattr(module, "execute_embedding_probe", successful_request)

    payload, exit_code = await module.run_probe(_args(execute=True))

    assert exit_code == 0
    assert payload["pass"] is True
    assert payload["result"]["dimensions"] == [3]
    assert payload["result"]["l2_norms"] == [1.0]
    assert payload["result"]["vectors_persisted"] is False
    serialized = json.dumps(payload, sort_keys=True)
    assert "private probe text" not in serialized
    assert "[1.0, 0.0, 0.0]" not in serialized


@pytest.mark.asyncio
async def test_probe_failure_is_typed_and_does_not_persist_message(monkeypatch):
    module = _load_script()
    from app.services.error_sanitizer import ExternalServiceError

    monkeypatch.setattr(
        module,
        "current_runtime_card",
        lambda: {
            "embedding_api_protocol": "openai",
            "embedding_model": "fixture-model",
            "embedding_dimensions": 3,
            "embedding_base_url_present": True,
            "embedding_api_key_present": True,
            "model_bridge_enabled": True,
            "model_fallback_enabled": False,
        },
    )
    monkeypatch.setattr(module, "sync_model_runtime_card", lambda: {"ok": True})

    async def failed_request(*_args, **_kwargs):
        error = ExternalServiceError(
            service="model_bridge",
            phase="http_json",
            status_code=502,
            error_code="provider_error",
            retryable=True,
        )
        wrapper = RuntimeError("private upstream response body")
        wrapper.__cause__ = error
        raise wrapper

    monkeypatch.setattr(module, "execute_embedding_probe", failed_request)

    payload, exit_code = await module.run_probe(_args(execute=True))

    assert exit_code == 2
    classification = payload["failure"]["classification"]
    assert classification["service"] == "model_bridge"
    assert classification["http_status"] == 502
    assert classification["retryable"] is True
    serialized = json.dumps(payload, sort_keys=True)
    assert "private upstream response body" not in serialized
    assert payload["failure"]["exception_message_persisted"] is False


@pytest.mark.asyncio
async def test_bridge_arm_preserves_only_safe_response_card(monkeypatch):
    module = _load_script()
    args = _args(execute=True)
    args.arm = "bridge"
    monkeypatch.setattr(
        module,
        "current_runtime_card",
        lambda: {
            "embedding_api_protocol": "openai",
            "embedding_model": "fixture-model",
            "embedding_dimensions": 3,
            "embedding_base_url_present": True,
            "embedding_api_key_present": True,
            "model_bridge_enabled": True,
            "model_fallback_enabled": False,
        },
    )
    monkeypatch.setattr(module, "sync_model_runtime_card", lambda: {"ok": True})

    async def bridge_response(*_args, **_kwargs):
        return {
            "http_status": 502,
            "content_type_json": True,
            "response_bytes": 76,
            "error_present": True,
            "error_code": "upstream_transport_error",
            "error_route": "embedding",
            "vector_count": 0,
            "dimensions": [],
            "expected_dimensions": 3,
            "dimension_match": False,
            "response_body_persisted": False,
            "vectors_persisted": False,
            "credentials_persisted": False,
        }

    monkeypatch.setattr(module, "execute_bridge_probe", bridge_response)

    payload, exit_code = await module.run_probe(args)

    assert exit_code == 2
    assert payload["arm"] == "bridge"
    assert payload["result"]["error_code"] == "upstream_transport_error"
    assert payload["result"]["response_body_persisted"] is False

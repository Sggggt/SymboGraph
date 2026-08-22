from __future__ import annotations

import hashlib
import json
import ssl
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.asyncio
async def test_chat_answer_payload_has_no_output_token_cap(monkeypatch, no_fallback_env):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    captured: dict = {}

    async def fake_post_chat(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None, purpose: str = "chat") -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        captured["resolve_ip"] = resolve_ip
        captured["purpose"] = purpose
        return {"choices": [{"message": {"content": "Grounded answer [1]"}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_chat)

    result = await ChatProvider().answer_question_with_meta(
        "解释贝叶斯网络",
        [
            {
                "document_title": "Bayes",
                "partition": "Chapter 1",
                "source_path": "/data/math/bayes-notes.md",
                "content": "贝叶斯网络使用有向边表达条件依赖，并结合证据更新后验。",
                "metadata": {
                    "document_id": "doc-internal",
                    "document_version_id": "version-internal",
                    "section_path": "Chapter 1 / Bayesian networks",
                    "page_range": [2, 3],
                    "char_span": [120, 260],
                },
            }
        ],
        [],
        max_factual_claims=6,
    )

    payload = captured["payload"]
    system_prompt = payload["messages"][0]["content"]
    assert result.answer == "Grounded answer [1]"
    assert payload["model"]
    assert "max_tokens" not in payload
    assert captured["purpose"] == "chat"
    assert "complete as the supplied evidence supports" in system_prompt
    normalized_prompt = system_prompt.lower()
    assert "keep the answer concise" not in normalized_prompt
    assert "answer concisely" not in normalized_prompt
    assert "keep clarification requests concise" in normalized_prompt
    user_prompt = payload["messages"][-1]["content"]
    assert "File: bayes-notes.md" in user_prompt
    assert "Source path: /data/math/bayes-notes.md" in user_prompt
    assert "Partition: Chapter 1" in user_prompt
    assert "Section: Chapter 1 / Bayesian networks" in user_prompt
    assert "Pages: 2-3" in user_prompt
    assert "Character span: 120-260" in user_prompt
    assert "doc-internal" not in user_prompt
    assert "version-internal" not in user_prompt
    assert "Hard answer-shape limit: use at most 6" in user_prompt
    assert "Hard answer-shape limit" not in system_prompt


@pytest.mark.asyncio
async def test_anthropic_grounded_answer_uses_claim_bounded_runtime_budget(
    monkeypatch,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="https://provider.example/ai",
        graph_base_url="https://provider.example/ai",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=False,
        chat_json_max_tokens=12_000,
        enable_model_fallback=False,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    provider = embeddings.ChatProvider()
    captured: dict = {}

    async def fake_post(payload: dict) -> str:
        captured.update(payload)
        return "Grounded answer."

    monkeypatch.setattr(provider, "_post_anthropic_sdk_text", fake_post)

    result = await provider.answer_question_with_meta(
        "What are the main concepts?",
        [{"document_title": "Source", "content": "Grounded context."}],
        [],
        max_factual_claims=6,
    )

    assert captured["max_tokens"] == 8192
    assert captured["thinking"] == {"type": "disabled"}
    assert result.output_token_budget == 8192
    assert result.output_token_budget_protocol_version == (
        "anthropic_grounded_answer_claim_bounded_output_budget_v1"
    )
    assert result.provider_call_audit is not None
    assert result.provider_call_audit["provider_response_persisted"] is False
    assert "private reasoning or chain-of-thought" in captured["messages"][-1][
        "content"
    ]


@pytest.mark.asyncio
async def test_chat_provider_keeps_chat_and_graph_endpoints_isolated(monkeypatch, no_fallback_env):
    from app.core import config
    from app.core.config import get_settings
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    monkeypatch.setenv("CHAT_API_KEY", "chat-key")
    monkeypatch.setenv("CHAT_BASE_URL", "https://chat.example.test/v1")
    monkeypatch.setenv("API_CHAT_BASE_URL", "https://chat.example.test/v1")
    monkeypatch.setenv("CHAT_MODEL", "chat-model")
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "false")
    monkeypatch.setenv("GRAPH_API_KEY", "graph-key")
    monkeypatch.setenv("GRAPH_BASE_URL", "https://graph.example.test/v1")
    monkeypatch.setenv("API_GRAPH_BASE_URL", "https://graph.example.test/v1")
    monkeypatch.setenv("GRAPH_MODEL", "graph-model")
    monkeypatch.setattr(
        config,
        "_read_workspace_env",
        lambda: {
            "MODEL_BRIDGE_ENABLED": "false",
            "CHAT_API_KEY": "chat-key",
            "CHAT_BASE_URL": "https://chat.example.test/v1",
            "CHAT_MODEL": "chat-model",
            "GRAPH_API_KEY": "graph-key",
            "GRAPH_BASE_URL": "https://graph.example.test/v1",
            "GRAPH_MODEL": "graph-model",
        },
    )
    get_settings.cache_clear()

    calls: list[dict] = []

    async def fake_post_chat(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None, purpose: str = "chat") -> dict:
        calls.append({"url": url, "payload": payload, "headers": headers, "purpose": purpose})
        return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_chat)

    await ChatProvider().classify_json("system", "user")
    await ChatProvider(purpose="graph").classify_json("system", "user")

    assert calls[0]["url"] == "https://chat.example.test/v1/chat/completions"
    assert calls[0]["payload"]["model"] == "chat-model"
    assert calls[0]["headers"]["Authorization"] == "Bearer chat-key"
    assert calls[0]["purpose"] == "chat"
    assert calls[1]["url"] == "https://graph.example.test/v1/chat/completions"
    assert calls[1]["payload"]["model"] == "graph-model"
    assert calls[1]["headers"]["Authorization"] == "Bearer graph-key"
    assert calls[1]["purpose"] == "graph"


@pytest.mark.asyncio
async def test_embedding_provider_syncs_model_bridge_before_external_request(monkeypatch, no_fallback_env):
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services import embeddings, runtime_settings
    from app.services.embeddings import EmbeddingProvider

    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    get_settings.cache_clear()
    sync_calls: list[str] = []
    captured: dict = {}

    def fake_sync_model_bridge_runtime_config(settings=None, env_entries=None, raise_on_error=True):
        sync_calls.append(settings.embedding_base_url)
        return {"attempted": True, "ok": True, "config_version": "bridge-version"}

    async def fake_post(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None, purpose: str = "embedding") -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["resolve_ip"] = resolve_ip
        captured["purpose"] = purpose
        return {"data": [{"embedding": [0.1] * 8}]}

    monkeypatch.setattr(runtime_settings, "sync_model_bridge_runtime_config", fake_sync_model_bridge_runtime_config)
    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post)

    result = await EmbeddingProvider().embed_texts_with_meta(["query"], text_type="query")

    assert result.external_called is True
    assert sync_calls == [model_bridge_client_base_url(8765)]
    assert captured["url"] == f"{model_bridge_client_base_url(8765)}/embeddings"
    assert captured["resolve_ip"] == "__none__"
    assert captured["purpose"] == "embedding"


@pytest.mark.asyncio
async def test_chat_provider_syncs_model_bridge_for_chat_only(monkeypatch, no_fallback_env):
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services import embeddings, runtime_settings
    from app.services.embeddings import ChatProvider

    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    get_settings.cache_clear()
    sync_calls: list[str] = []
    calls: list[tuple[str, str]] = []

    def fake_sync_model_bridge_runtime_config(settings=None, env_entries=None, raise_on_error=True):
        sync_calls.append(settings.chat_base_url)
        return {"attempted": True, "ok": True, "config_version": "bridge-version"}

    async def fake_post(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None, purpose: str = "chat") -> dict:
        calls.append((url, purpose))
        return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

    monkeypatch.setattr(runtime_settings, "sync_model_bridge_runtime_config", fake_sync_model_bridge_runtime_config)
    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post)

    await ChatProvider().classify_json("system", "user")
    await ChatProvider(purpose="graph").classify_json("system", "user")

    assert sync_calls == [model_bridge_client_base_url(8765)]
    assert calls[0] == (f"{model_bridge_client_base_url(8765)}/chat/completions", "chat")
    assert calls[1] == ("https://graph.invalid/v1/chat/completions", "graph")


@pytest.mark.asyncio
async def test_openai_compatible_json_retries_transient_transport_errors(monkeypatch, no_fallback_env):
    from app.services import embeddings

    calls = 0
    sleeps: list[float] = []

    def fake_resolve(hostname: str, port: int, timeout: float) -> str:
        assert (hostname, port, timeout) == ("provider.example", 443, 1.0)
        return "1.1.1.1"

    def fake_post(url, payload, headers, timeout, resolve_ip):
        nonlocal calls
        calls += 1
        assert resolve_ip == "1.1.1.1"
        if calls < embeddings.MODEL_REQUEST_MAX_ATTEMPTS:
            raise httpx.ConnectError("", request=httpx.Request("POST", url))
        return {"ok": True}

    async def fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(embeddings, "_resolve_public_provider_ip", fake_resolve)
    monkeypatch.setattr(embeddings, "_post_json_with_pinned_resolve", fake_post)
    monkeypatch.setattr(embeddings.asyncio, "sleep", fake_sleep)

    result = await embeddings.post_openai_compatible_json(
        "https://provider.example/v1/chat/completions",
        {"model": "unit"},
        {"Authorization": "Bearer redacted"},
        timeout=1,
        purpose="chat",
    )

    assert result == {"ok": True}
    assert calls == embeddings.MODEL_REQUEST_MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_direct_provider_transport_has_no_legacy_or_system_dns_path():
    from app.services import embeddings

    source = Path(embeddings.__file__).read_text(encoding="utf-8")
    forbidden = (
        "socket.getaddrinfo",
        "subprocess",
        "tempfile",
        "NamedTemporaryFile",
        "symbograph-openai",
        "curl.exe",
        " curl ",
    )
    assert [token for token in forbidden if token in source] == []


def test_provider_resolution_uses_bounded_pinned_doh_without_system_dns(monkeypatch):
    from app.services import embeddings

    calls: list[dict] = []

    def fail_system_dns(*args, **kwargs):
        raise AssertionError("system DNS must never be used for provider resolution")

    def fake_doh(**kwargs):
        calls.append(kwargs)
        return "8.8.8.8"

    monkeypatch.setattr(embeddings.socket, "getaddrinfo", fail_system_dns)
    monkeypatch.setattr(embeddings, "_query_doh_public_a", fake_doh)

    result = embeddings._resolve_public_provider_ip("provider.example", 443, 3.0)

    assert result == "8.8.8.8"
    assert len(calls) == 1
    assert calls[0] == {
        "resolver_hostname": "dns.alidns.com",
        "resolver_ip": "223.5.5.5",
        "resolver_path": "/resolve",
        "provider_hostname": "provider.example",
        "timeout": 3.0,
    }


def test_provider_resolution_rejects_private_literal_without_doh(monkeypatch):
    from app.services import embeddings

    monkeypatch.setattr(
        embeddings,
        "_query_doh_public_a",
        lambda **kwargs: pytest.fail("literal IP must not invoke DoH"),
    )
    with pytest.raises(RuntimeError, match="public unicast"):
        embeddings._resolve_public_provider_ip("127.0.0.1", 443)


class _FakeHTTPResponse:
    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None):
        self.status = status
        self._body = body
        self._headers = headers or {
            "Content-Type": "application/json",
            "Content-Encoding": "identity",
            "Content-Length": str(len(body)),
        }

    def getheader(self, name: str):
        return self._headers.get(name)

    def read(self, limit: int):
        return self._body[:limit]


def _install_fake_pinned_connection(monkeypatch, embeddings, response):
    captured: dict = {}

    class FakeConnection:
        def __init__(self, hostname, connect_ip, port, *, timeout, context):
            captured.update(
                hostname=hostname,
                connect_ip=connect_ip,
                port=port,
                timeout=timeout,
                context=context,
            )

        def request(self, method, target, body=None, headers=None):
            captured.update(method=method, target=target, body=body, headers=headers)

        def getresponse(self):
            return response

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(embeddings, "_PinnedHTTPSConnection", FakeConnection)
    return captured


def test_doh_query_is_pinned_verified_bounded_and_public(monkeypatch):
    from app.services import embeddings

    body = json.dumps(
        {
            "Status": 0,
            "TC": False,
            "Answer": [
                {"name": "provider.example.", "type": 5, "data": "edge.example."},
                {"name": "edge.example.", "type": 1, "data": "8.8.8.8"},
            ],
        }
    ).encode()
    response = _FakeHTTPResponse(
        200,
        body,
        {
            "Content-Type": "application/dns-json; charset=utf-8",
            "Content-Encoding": "identity",
            "Content-Length": str(len(body)),
        },
    )
    captured = _install_fake_pinned_connection(monkeypatch, embeddings, response)

    result = embeddings._query_doh_public_a(
        resolver_hostname="cloudflare-dns.com",
        resolver_ip="1.1.1.1",
        resolver_path="/dns-query",
        provider_hostname="provider.example",
        timeout=2.0,
    )

    assert result == "8.8.8.8"
    assert captured["hostname"] == "cloudflare-dns.com"
    assert captured["connect_ip"] == "1.1.1.1"
    assert captured["port"] == 443
    assert captured["context"].verify_mode == ssl.CERT_REQUIRED
    assert captured["context"].check_hostname is True
    assert captured["method"] == "GET"
    assert captured["target"] == "/dns-query?name=provider.example&type=A"
    assert captured["headers"] == {
        "Accept": "application/dns-json",
        "Accept-Encoding": "identity",
    }
    assert captured["closed"] is True


@pytest.mark.asyncio
async def test_anthropic_protocol_uses_model_bridge_without_auth_drift(
    monkeypatch,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="openai",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="http://host.docker.internal:8765",
        graph_base_url="https://graph.example/v1",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=True,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    # A stale Compose/process value must not override the active Settings
    # route after runtime settings have selected the bridge.
    monkeypatch.setenv("CHAT_BASE_URL", "https://provider.example/ai")
    sync_calls: list[str] = []
    monkeypatch.setattr(
        embeddings,
        "_sync_model_bridge_for_model_io",
        lambda active_settings: sync_calls.append(active_settings.chat_base_url),
    )
    captured: dict = {}

    class FakeMessages:
        async def create(self, **_payload):
            return SimpleNamespace(
                model_dump=lambda **_kwargs: {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 40,
                        "cache_read_input_tokens": 80,
                    },
                }
            )

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = FakeMessages()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(embeddings, "AsyncAnthropic", FakeAsyncAnthropic)

    provider = embeddings.ChatProvider()
    text = await provider._post_chat_text(
        {
            "model": "unit-chat-model",
            "messages": [{"role": "user", "content": "question"}],
        }
    )

    assert text == "OK"
    assert provider.provider_call_audit()["usage"] == {
        "protocol_version": embeddings.PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION,
        "api_protocol": "anthropic",
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": None,
        "cache_creation_input_tokens": 40,
        "cache_read_input_tokens": 80,
        "cache_hit": True,
        "cache_write": True,
        "token_accounting_mode": (
            "provider_reported_anthropic_fields_no_cross_field_inference_v1"
        ),
        "usage_present": True,
        "provider_response_persisted": False,
    }
    assert sync_calls == ["http://host.docker.internal:8765"]
    assert provider.base_url == "http://host.docker.internal:8765"
    assert captured["auth_token"] == "unit-chat-key"
    assert captured["base_url"] == "http://host.docker.internal:8765"


def test_anthropic_graph_route_ignores_stale_process_base_url(monkeypatch):
    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="http://host.docker.internal:8765",
        graph_base_url="https://graph.example/ai",
        chat_resolve_ip="__none__",
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=True,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    monkeypatch.setenv("GRAPH_BASE_URL", "https://graph.example.test/v1")

    provider = embeddings.ChatProvider(purpose="graph")

    assert provider.base_url == "https://graph.example/ai"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"Status": 0, "TC": False, "Answer": [{"type": 1, "data": "127.0.0.1"}]},
            "non-public A record",
        ),
        (
            {"Status": 0, "Answer": [{"type": 1, "data": "8.8.8.8"}]},
            "complete successful",
        ),
        ({"Status": 0, "TC": True, "Answer": []}, "complete successful"),
        ({"Status": False, "Answer": []}, "complete successful"),
        ({"Status": 2, "Answer": []}, "complete successful"),
        ({"Status": 0, "TC": False, "Answer": "invalid"}, "record bound"),
        (
            {"Status": 0, "TC": False, "Answer": [{"type": "1", "data": "8.8.8.8"}]},
            "type was malformed",
        ),
        (
            {"Status": 0, "TC": False, "Answer": [{"type": 0, "data": "8.8.8.8"}]},
            "type was malformed",
        ),
        (
            {"Status": 0, "TC": False, "Answer": [{"type": 65536, "data": "8.8.8.8"}]},
            "type was malformed",
        ),
        (
            {"Status": 0, "TC": False, "Answer": [{"type": 1, "data": "2606:4700:4700::1111"}]},
            "non-public A record",
        ),
        (
            {"Status": 0, "TC": False, "Answer": [{"type": 1, "data": " 8.8.8.8 "}]},
            "data was malformed",
        ),
    ],
)
def test_doh_query_rejects_unsafe_or_malformed_answers(monkeypatch, payload, message):
    from app.services import embeddings

    body = json.dumps(payload).encode()
    response = _FakeHTTPResponse(
        200,
        body,
        {
            "Content-Type": "application/dns-json",
            "Content-Length": str(len(body)),
        },
    )
    _install_fake_pinned_connection(monkeypatch, embeddings, response)

    with pytest.raises(RuntimeError, match=message):
        embeddings._query_doh_public_a(
            resolver_hostname="cloudflare-dns.com",
            resolver_ip="1.1.1.1",
            resolver_path="/dns-query",
            provider_hostname="provider.example",
            timeout=2.0,
        )


def test_doh_query_rejects_redirect_without_consuming_location(monkeypatch):
    from app.services import embeddings

    sentinel = "doh-redirect-secret.invalid/path"
    response = _FakeHTTPResponse(
        302,
        b"",
        {"Location": f"https://{sentinel}"},
    )
    captured = _install_fake_pinned_connection(monkeypatch, embeddings, response)

    with pytest.raises(RuntimeError, match="redirect was rejected") as error:
        embeddings._query_doh_public_a(
            resolver_hostname="cloudflare-dns.com",
            resolver_ip="1.1.1.1",
            resolver_path="/dns-query",
            provider_hostname="provider.example",
            timeout=2.0,
        )

    assert sentinel not in str(error.value)
    assert captured["closed"] is True


def test_doh_query_enforces_body_and_answer_count_bounds(monkeypatch):
    from app.services import embeddings

    oversized_answers = {
        "Status": 0,
        "TC": False,
        "Answer": [
            {"type": 1, "data": "8.8.8.8"}
            for _ in range(embeddings.MAX_DOH_ANSWER_RECORDS + 1)
        ],
    }
    body = json.dumps(oversized_answers).encode()
    _install_fake_pinned_connection(monkeypatch, embeddings, _FakeHTTPResponse(200, body))
    with pytest.raises(RuntimeError, match="record bound"):
        embeddings._query_doh_public_a(
            resolver_hostname="cloudflare-dns.com",
            resolver_ip="1.1.1.1",
            resolver_path="/dns-query",
            provider_hostname="provider.example",
            timeout=2.0,
        )

    monkeypatch.setattr(embeddings, "MAX_DOH_RESPONSE_BODY_BYTES", 8)
    body = b'{"Status":0}'
    _install_fake_pinned_connection(monkeypatch, embeddings, _FakeHTTPResponse(200, body))
    with pytest.raises(RuntimeError, match="hard byte bound"):
        embeddings._query_doh_public_a(
            resolver_hostname="cloudflare-dns.com",
            resolver_ip="1.1.1.1",
            resolver_path="/dns-query",
            provider_hostname="provider.example",
            timeout=2.0,
        )


def test_direct_provider_pins_ip_but_preserves_hostname_tls_and_headers(monkeypatch):
    from app.services import embeddings

    body = b'{"ok":true}'
    response = _FakeHTTPResponse(200, body)
    captured = _install_fake_pinned_connection(monkeypatch, embeddings, response)

    result = embeddings._post_json_with_pinned_resolve(
        "https://provider.example/v1/chat/completions",
        {"model": "unit"},
        {"Authorization": "Bearer secret", "Content-Type": "application/json"},
        4.0,
        "1.1.1.1",
    )

    assert result == {"ok": True}
    assert captured["hostname"] == "provider.example"
    assert captured["connect_ip"] == "1.1.1.1"
    assert captured["context"].verify_mode == ssl.CERT_REQUIRED
    assert captured["context"].check_hostname is True
    assert captured["method"] == "POST"
    assert captured["target"] == "/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }
    assert captured["closed"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_bridge_enabled", "url", "purpose"),
    [
        (True, "https://provider.example/v1/chat/completions", "graph"),
        (False, "https://provider.example/v1/embeddings", "embedding"),
    ],
)
async def test_direct_transport_does_not_revalidate_expanded_headers(
    monkeypatch,
    model_bridge_enabled,
    url,
    purpose,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        model_bridge_enabled=model_bridge_enabled,
        model_bridge_port=8765,
    )
    body = b'{"ok":true}'
    captured = _install_fake_pinned_connection(
        monkeypatch,
        embeddings,
        _FakeHTTPResponse(200, body),
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)

    result = await embeddings.post_openai_compatible_json(
        url,
        {"model": "unit"},
        {"Authorization": "Bearer secret", "Content-Type": "application/json"},
        timeout=2.0,
        resolve_ip="1.1.1.1",
        purpose=purpose,
    )

    assert result == {"ok": True}
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }
    assert captured["closed"] is True


@pytest.mark.asyncio
async def test_direct_transport_rejects_extra_header_before_connection(monkeypatch):
    from app.services import embeddings

    settings = SimpleNamespace(model_bridge_enabled=False, model_bridge_port=8765)
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    monkeypatch.setattr(
        embeddings,
        "_PinnedHTTPSConnection",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported headers must fail before connection construction"
        ),
    )

    with pytest.raises(ValueError, match="unsupported header"):
        await embeddings.post_openai_compatible_json(
            "https://provider.example/v1/embeddings",
            {"model": "unit"},
            {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
                "X-Forward-Me": "no",
            },
            timeout=2.0,
            resolve_ip="1.1.1.1",
            purpose="embedding",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("app_env", ["development", "production"])
async def test_anthropic_provider_always_uses_official_sdk_with_bearer_auth(
    monkeypatch,
    app_env,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        app_env=app_env,
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="https://provider.example/ai",
        graph_base_url="https://provider.example/ai",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=False,
        mid_concept_extraction_max_tokens_per_batch=2400,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    captured: dict = {}

    class FakeMessages:
        async def create(self, **payload):
            captured["payload"] = payload
            return SimpleNamespace(
                model_dump=lambda **_kwargs: {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "cache_creation_input_tokens": 40,
                        "cache_read_input_tokens": 80,
                    },
                }
            )

    class FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.messages = FakeMessages()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            captured["closed"] = True

    monkeypatch.setattr(embeddings, "AsyncAnthropic", FakeAsyncAnthropic)
    monkeypatch.setattr(
        embeddings,
        "post_anthropic_json",
        lambda *_args, **_kwargs: pytest.fail(
            "Anthropic protocol must always use the official SDK"
        ),
    )

    provider = embeddings.ChatProvider()
    text = await provider._post_chat_text(
        {
            "model": "unit-chat-model",
            "messages": [
                {"role": "system", "content": "system rule"},
                {"role": "user", "content": "question"},
            ],
            "temperature": 0.0,
        }
    )

    assert text == "OK"
    assert captured["client"] == {
        "auth_token": "unit-chat-key",
        "base_url": "https://provider.example/ai",
        "timeout": 17.0,
        "max_retries": 0,
    }
    assert captured["payload"] == {
        "model": "unit-chat-model",
        "max_tokens": embeddings.ANTHROPIC_DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": "question"}],
        "system": [
            {
                "type": "text",
                "text": "system rule",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "temperature": 0.0,
    }
    audit = provider.provider_call_audit()
    assert audit["prompt_cache"] == {
        "protocol_version": embeddings.PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION,
        "api_protocol": "anthropic",
        "cache_mode": "anthropic_explicit_ephemeral",
        "cacheable_system_prompt_present": True,
        "cacheable_system_prompt_sha256": hashlib.sha256(
            b"system rule"
        ).hexdigest(),
        "cacheable_system_prompt_utf8_bytes": len(b"system rule"),
        "provider_response_persisted": False,
    }
    assert audit["usage"]["cache_hit"] is True
    assert audit["usage"]["cache_read_input_tokens"] == 80
    assert captured["closed"] is True


@pytest.mark.asyncio
async def test_anthropic_graph_json_budget_is_derived_from_concept_input_budget(
    monkeypatch,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="https://provider.example/ai",
        graph_base_url="https://provider.example/ai",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=False,
        mid_concept_extraction_max_tokens_per_batch=2400,
        enable_model_fallback=False,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    provider = embeddings.ChatProvider(purpose="graph")
    captured: dict = {}

    async def fake_post(payload):
        captured.update(payload)
        return {"concepts": []}

    monkeypatch.setattr(provider, "_post_chat_json_with_response_format_fallback", fake_post)

    assert await provider.classify_json("system", "user") == {"concepts": []}
    assert captured["max_tokens"] == 9600
    assert captured["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_anthropic_chat_json_budget_is_explicit_and_hot_configurable(
    monkeypatch,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="https://provider.example/ai",
        graph_base_url="https://provider.example/ai",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=False,
        chat_json_max_tokens=12_000,
        enable_model_fallback=False,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    provider = embeddings.ChatProvider(purpose="chat")
    captured: dict = {}

    async def fake_post(payload):
        captured.update(payload)
        return {"typed_actions": []}

    monkeypatch.setattr(
        provider,
        "_post_chat_json_with_response_format_fallback",
        fake_post,
    )

    assert await provider.classify_json("system", "user") == {
        "typed_actions": []
    }
    assert captured["max_tokens"] == 12_000
    assert captured["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_structured_json_component_budget_only_tightens_runtime_cap(
    monkeypatch,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="https://provider.example/ai",
        graph_base_url="https://provider.example/ai",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=False,
        chat_json_max_tokens=12_000,
        enable_model_fallback=False,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    provider = embeddings.ChatProvider(purpose="chat")
    captured: dict = {}

    async def fake_post(payload):
        captured.update(payload)
        return {"verdict": "sufficient"}

    monkeypatch.setattr(
        provider,
        "_post_chat_json_with_response_format_fallback",
        fake_post,
    )

    assert await provider.classify_json_bounded(
        "system",
        "user",
        max_tokens=4096,
    ) == {"verdict": "sufficient"}
    assert captured["max_tokens"] == 4096
    assert captured["thinking"] == {"type": "disabled"}

    settings.chat_json_max_tokens = 1024
    captured.clear()
    await provider.classify_json_bounded(
        "system",
        "user",
        max_tokens=4096,
    )
    assert captured["max_tokens"] == 1024


@pytest.mark.asyncio
async def test_structured_json_budget_helper_preserves_local_test_adapter():
    from app.services.embeddings import classify_json_with_budget

    class LocalAdapter:
        calls = 0

        async def classify_json(self, system_prompt, user_prompt, fallback=None):
            self.calls += 1
            return {"ok": True}

    adapter = LocalAdapter()
    assert await classify_json_with_budget(
        adapter,
        system_prompt="system",
        user_prompt="user",
        fallback=None,
        max_tokens=1024,
    ) == {"ok": True}
    assert adapter.calls == 1


def test_anthropic_payload_accepts_only_disabled_thinking_for_structured_graph():
    from app.services import embeddings

    provider = object.__new__(embeddings.ChatProvider)
    base = {
        "model": "unit-model",
        "messages": [{"role": "user", "content": "question"}],
        "thinking": {"type": "disabled"},
    }

    assert provider._anthropic_messages_payload(base)["thinking"] == {
        "type": "disabled"
    }
    with pytest.raises(ValueError, match="require disabled thinking"):
        provider._anthropic_messages_payload(
            {**base, "thinking": {"type": "enabled", "budget_tokens": 1024}}
        )


def test_chat_json_parser_fails_closed_without_leaking_provider_text():
    from app.services import embeddings

    provider = object.__new__(embeddings.ChatProvider)
    sentinel = "SENTINEL_PROVIDER_RESPONSE_MUST_NOT_ESCAPE"

    with pytest.raises(RuntimeError) as malformed:
        provider._parse_json_object(f"not-json {sentinel}")
    message = str(malformed.value)
    assert "provider_json_text_shape_v1" in message
    assert "sha256" in message
    assert sentinel not in message

    with pytest.raises(embeddings.ProviderJSONShapeError) as non_object:
        provider._parse_json_object('[{"concepts": []}]')
    assert non_object.value.diagnostics["error_code"] == "json_root_not_object"
    assert non_object.value.diagnostics["json_type"] == "list"

    assert provider._parse_json_object('{"concepts": []}') == {"concepts": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "expected_error_code"),
    [
        ("max_tokens", "incomplete_max_tokens"),
        ("refusal", "provider_refusal"),
        (None, "invalid_stop_reason"),
    ],
)
async def test_anthropic_sdk_rejects_incomplete_or_unknown_completion(
    monkeypatch,
    stop_reason,
    expected_error_code,
):
    from app.services import embeddings
    from app.services.error_sanitizer import ExternalServiceError

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="https://provider.example/ai",
        graph_base_url="https://provider.example/ai",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=17,
        model_bridge_enabled=False,
        mid_concept_extraction_max_tokens_per_batch=2400,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)

    class FakeMessages:
        async def create(self, **_payload):
            return SimpleNamespace(
                model_dump=lambda **_kwargs: {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "partial"}],
                    "stop_reason": stop_reason,
                }
            )

    class FakeAsyncAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    monkeypatch.setattr(embeddings, "AsyncAnthropic", FakeAsyncAnthropic)

    with pytest.raises(ExternalServiceError) as captured:
        await embeddings.ChatProvider(purpose="graph")._post_chat_text(
            {
                "model": "unit-graph-model",
                "messages": [{"role": "user", "content": "question"}],
            }
        )

    assert captured.value.error_code == expected_error_code
    assert captured.value.retryable is False


@pytest.mark.asyncio
async def test_anthropic_sdk_retries_typed_connection_error_without_message_leak(
    monkeypatch,
):
    import anthropic
    import httpx

    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol="anthropic",
        graph_api_protocol="anthropic",
        chat_api_key="unit-chat-key",
        graph_api_key="unit-graph-key",
        chat_base_url="https://provider.example/ai",
        graph_base_url="https://provider.example/ai",
        chat_resolve_ip=None,
        graph_resolve_ip=None,
        chat_model="unit-chat-model",
        graph_model="unit-graph-model",
        model_request_timeout_seconds=1.0,
        model_bridge_enabled=False,
        model_bridge_port=8765,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    monkeypatch.setattr(embeddings.asyncio, "sleep", lambda *_args: _async_none())
    attempts = 0

    class FakeMessages:
        async def create(self, **_payload):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise anthropic.APIConnectionError(
                    message="private-transport-message-must-not-persist",
                    request=httpx.Request("POST", "https://provider.example/ai"),
                )
            return SimpleNamespace(
                model_dump=lambda **_kwargs: {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn",
                }
            )

    class FakeAsyncAnthropic:
        def __init__(self, **_kwargs):
            self.messages = FakeMessages()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    async def _async_none():
        return None

    monkeypatch.setattr(embeddings, "AsyncAnthropic", FakeAsyncAnthropic)

    text = await embeddings.ChatProvider(purpose="graph")._post_chat_text(
        {
            "model": "unit-graph-model",
            "messages": [{"role": "user", "content": "question"}],
        }
    )

    assert text == "OK"
    assert attempts == 2


@pytest.mark.parametrize(
    ("chat_protocol", "graph_protocol"),
    [
        ("openai", "openai"),
        ("openai", "anthropic"),
        ("anthropic", "openai"),
        ("anthropic", "anthropic"),
    ],
)
def test_provider_protocol_matrix_keeps_chat_graph_and_embedding_isolated(
    monkeypatch,
    chat_protocol,
    graph_protocol,
):
    from app.services import embeddings

    settings = SimpleNamespace(
        chat_api_protocol=chat_protocol,
        graph_api_protocol=graph_protocol,
        embedding_api_protocol="openai",
        chat_api_key="chat-key",
        graph_api_key="graph-key",
        embedding_api_key="embedding-key",
        chat_base_url="https://chat.example.test",
        graph_base_url="https://graph.example.test",
        embedding_base_url="https://embedding.example.test/v1",
        chat_resolve_ip="1.1.1.1",
        graph_resolve_ip="2.2.2.2",
        embedding_resolve_ip="3.3.3.3",
        chat_model="chat-model",
        graph_model="graph-model",
        embedding_model="embedding-model",
        embedding_dimensions=8,
        enable_model_fallback=False,
    )
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)

    chat = embeddings.ChatProvider("chat")
    graph = embeddings.ChatProvider("graph")
    vector = embeddings.EmbeddingProvider()

    assert (chat.api_protocol, chat.api_key, chat.base_url, chat.model) == (
        chat_protocol,
        "chat-key",
        "https://chat.example.test",
        "chat-model",
    )
    assert (graph.api_protocol, graph.api_key, graph.base_url, graph.model) == (
        graph_protocol,
        "graph-key",
        "https://graph.example.test",
        "graph-model",
    )
    assert vector.embedding_model == "embedding-model"
    assert vector.embedding_dimensions == 8
    assert vector.embedding_api_protocol == "openai"


@pytest.mark.asyncio
async def test_anthropic_direct_transport_pins_tls_and_exact_headers(monkeypatch):
    from app.services import embeddings

    settings = SimpleNamespace(model_bridge_enabled=False, model_bridge_port=8765)
    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    captured = _install_fake_pinned_connection(
        monkeypatch,
        embeddings,
        _FakeHTTPResponse(
            200,
            b'{"content":[{"type":"text","text":"ok"}]}',
        ),
    )

    result = await embeddings.post_anthropic_json(
        "https://provider.example/v1/messages",
        {
            "model": "unit-model",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "question"}],
        },
        {
            "X-Api-Key": "unit-key",
            "Anthropic-Version": embeddings.ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        timeout=2.0,
        resolve_ip="1.1.1.1",
        purpose="graph",
    )

    assert result["content"][0]["text"] == "ok"
    assert captured["hostname"] == "provider.example"
    assert captured["connect_ip"] == "1.1.1.1"
    assert captured["context"].verify_mode == ssl.CERT_REQUIRED
    assert captured["context"].check_hostname is True
    assert captured["target"] == "/v1/messages"
    assert captured["headers"] == {
        "X-Api-Key": "unit-key",
        "Anthropic-Version": embeddings.ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }
    assert captured["closed"] is True


@pytest.mark.asyncio
async def test_anthropic_transport_rejects_extra_header_before_connection(
    monkeypatch,
):
    from app.services import embeddings

    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: SimpleNamespace(model_bridge_enabled=False, model_bridge_port=8765),
    )
    monkeypatch.setattr(
        embeddings,
        "_PinnedHTTPSConnection",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported Anthropic headers must fail before connection construction"
        ),
    )
    with pytest.raises(ValueError, match="unsupported header"):
        await embeddings.post_anthropic_json(
            "https://provider.example/v1/messages",
            {
                "model": "unit-model",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "question"}],
            },
            {
                "X-Api-Key": "unit-key",
                "Anthropic-Version": embeddings.ANTHROPIC_VERSION,
                "Content-Type": "application/json",
                "Authorization": "must-not-cross-protocol",
            },
            timeout=2.0,
            resolve_ip="1.1.1.1",
            purpose="graph",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "error", "error": {"type": "unit"}, "content": [{"type": "text", "text": "must-not-pass"}]},
        {"type": "message", "role": "user", "content": [{"type": "text", "text": "must-not-pass"}]},
        {"type": "message", "role": "assistant", "content": []},
        {"type": "message", "role": "assistant", "content": [{"type": "tool_use", "id": "unit"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "text", "text": ""}]},
    ],
)
def test_anthropic_response_contract_fails_closed(payload):
    from app.services import embeddings

    provider = object.__new__(embeddings.ChatProvider)
    with pytest.raises(embeddings.ProviderJSONShapeError) as error:
        provider._normalize_anthropic_content(payload)
    assert error.value.diagnostics["protocol_version"] == (
        embeddings.ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION
    )
    assert error.value.diagnostics["error_code"]
    assert set(error.value.diagnostics) <= {
        "protocol_version",
        "error_code",
        "field_path",
        "json_type",
    }


def test_anthropic_response_contract_ignores_thinking_and_requires_text():
    from app.services import embeddings

    provider = object.__new__(embeddings.ChatProvider)
    payload = {
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "private reasoning"},
            {"type": "text", "text": "public answer"},
        ],
    }

    assert provider._normalize_anthropic_content(payload) == "public answer"

    with pytest.raises(embeddings.ProviderJSONShapeError) as error:
        provider._normalize_anthropic_content(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "only reasoning"}],
            }
        )
    assert error.value.diagnostics["error_code"] == "empty_text_content"


def test_provider_controlled_error_fields_and_refusal_never_escape():
    from app.services import embeddings
    from app.services.error_sanitizer import public_exception_message

    sentinel = "SENTINEL_PROVIDER_SECRET_7bb83b7291a14db0"
    body = json.dumps(
        {
            "error": {
                "code": sentinel,
                "type": sentinel,
                "param": sentinel,
                "message": f"unsupported parameter response_format {sentinel}",
            }
        }
    )
    error = embeddings._external_provider_error_from_body(
        body,
        phase="unit_provider_error",
    )

    assert error.error_code == "provider_error"
    assert error.unsupported_parameters == {"response_format"}
    assert sentinel not in public_exception_message(error)
    assert sentinel not in str(error)

    provider = object.__new__(embeddings.ChatProvider)
    with pytest.raises(RuntimeError) as refusal:
        provider._normalize_chat_content(
            {
                "choices": [
                    {
                        "message": {
                            "refusal": sentinel,
                            "content": None,
                        }
                    }
                ]
            }
        )
    assert str(refusal.value) == "Chat response contained a refusal"
    assert sentinel not in str(refusal.value)


@pytest.mark.asyncio
async def test_anthropic_bridge_redirect_error_is_not_retried(monkeypatch):
    from app.services import embeddings
    from app.services.error_sanitizer import ExternalServiceError

    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: SimpleNamespace(model_bridge_enabled=True, model_bridge_port=8765),
    )
    calls = 0

    class FakeResponse:
        status_code = 502
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "identity",
        }

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_bytes(self):
            yield b'{"error":{"code":"upstream_redirect_rejected","status":302}}'

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return FakeResponse()

    async def fail_sleep(_delay):
        pytest.fail("a bridge redirect must not be retried")

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(embeddings.asyncio, "sleep", fail_sleep)

    with pytest.raises(ExternalServiceError) as captured:
        await embeddings.post_anthropic_json(
            "http://host.docker.internal:8765/v1/messages",
            {
                "model": "unit-model",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "question"}],
            },
            {
                "X-Api-Key": "unit-key",
                "Anthropic-Version": embeddings.ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            timeout=2.0,
            purpose="chat",
        )

    assert calls == 1
    assert captured.value.error_code == "upstream_redirect_rejected"
    assert captured.value.retryable is False


def test_direct_provider_enforces_response_content_length_before_read(monkeypatch):
    from app.services import embeddings

    monkeypatch.setattr(embeddings, "MAX_OPENAI_RESPONSE_BODY_BYTES", 8)
    response = _FakeHTTPResponse(200, b'{"ok":true}')
    captured = _install_fake_pinned_connection(monkeypatch, embeddings, response)

    with pytest.raises(RuntimeError, match="hard byte bound"):
        embeddings._post_json_with_pinned_resolve(
            "https://provider.example/v1/embeddings",
            {"model": "unit"},
            {"Authorization": "Bearer secret"},
            4.0,
            "1.1.1.1",
        )

    assert captured["closed"] is True


def test_direct_provider_rejects_redirect_without_disclosing_location(monkeypatch):
    from app.services import embeddings
    from app.services.error_sanitizer import public_exception_message

    sentinel = "redirect-secret.invalid/credential"
    response = _FakeHTTPResponse(
        307,
        b"",
        {"Location": f"https://{sentinel}"},
    )
    _install_fake_pinned_connection(monkeypatch, embeddings, response)

    with pytest.raises(Exception) as captured:
        embeddings._post_json_with_pinned_resolve(
            "https://provider.example/v1/embeddings",
            {"model": "unit"},
            {"Authorization": "Bearer secret"},
            4.0,
            "1.1.1.1",
        )

    assert sentinel not in str(captured.value)
    assert sentinel not in public_exception_message(captured.value)


@pytest.mark.parametrize(
    (
        "status",
        "content_type",
        "content_encoding",
        "expected_error_code",
        "expected_retryable",
    ),
    [
        (200, "text/html; charset=utf-8", "identity", "non_json_content_type", False),
        (400, "text/html; charset=utf-8", "identity", "non_json_content_type", False),
        (429, "text/html; charset=utf-8", "identity", "non_json_content_type", True),
        (502, "text/html; charset=utf-8", "identity", "non_json_content_type", True),
        (200, "application/json", "gzip", "unsupported_content_encoding", False),
        (503, "application/json", "gzip", "unsupported_content_encoding", True),
    ],
)
def test_direct_provider_non_json_headers_preserve_status_without_reading_body(
    monkeypatch,
    status,
    content_type,
    content_encoding,
    expected_error_code,
    expected_retryable,
):
    from app.services import embeddings
    from app.services.error_sanitizer import ExternalServiceError

    class NonJsonResponse(_FakeHTTPResponse):
        def __init__(self):
            super().__init__(
                status,
                b"must-not-be-read",
                {
                    "Content-Type": content_type,
                    "Content-Encoding": content_encoding,
                },
            )
            self.read_called = False

        def read(self, limit: int):
            self.read_called = True
            return super().read(limit)

    response = NonJsonResponse()
    captured = _install_fake_pinned_connection(monkeypatch, embeddings, response)

    with pytest.raises(ExternalServiceError) as error:
        embeddings._post_json_with_pinned_resolve(
            "https://provider.example/v1/chat/completions",
            {"model": "unit"},
            {"Authorization": "Bearer secret"},
            4.0,
            "1.1.1.1",
        )

    assert error.value.status_code == status
    assert error.value.error_code == expected_error_code
    assert error.value.retryable is expected_retryable
    assert response.read_called is False
    assert captured["closed"] is True


@pytest.mark.parametrize(
    ("body", "content_type", "content_encoding", "message"),
    [
        (b"[]", "application/json", "identity", "one JSON object"),
        (b"not-json", "application/json", "identity", "not valid JSON"),
        (b"\xff", "application/json", "identity", "not valid JSON"),
        (b'{"value":NaN}', "application/json", "identity", "not valid JSON"),
        (b'{"value":1,"value":2}', "application/json", "identity", "not valid JSON"),
        (b"{}", "text/html", "identity", "Content-Type was not JSON"),
        (b"{}", "application/json", "gzip", "content encoding"),
    ],
)
def test_provider_response_validation_fails_closed(body, content_type, content_encoding, message):
    from app.services import embeddings

    with pytest.raises(RuntimeError, match=message):
        embeddings._validated_json_response_body(
            body,
            content_type=content_type,
            content_encoding=content_encoding,
        )


def test_provider_request_validation_fails_before_network(monkeypatch, no_fallback_env):
    from app.services import embeddings

    monkeypatch.setattr(embeddings, "MAX_OPENAI_REQUEST_BODY_BYTES", 8)
    monkeypatch.setattr(
        embeddings,
        "_post_json_with_pinned_resolve",
        lambda *args, **kwargs: pytest.fail("network transport must not be reached"),
    )

    with pytest.raises(RuntimeError, match="request exceeded"):
        embeddings._validated_openai_request_body({"payload": "too-large"})
    with pytest.raises(ValueError, match="unsupported header"):
        embeddings._restricted_openai_headers(
            {"Authorization": "Bearer secret", "X-Forward-Me": "no"}
        )
    with pytest.raises(RuntimeError, match="Authorization"):
        embeddings._restricted_openai_headers(
            {"Authorization": "Bearer secret\r\nX-Evil: yes"}
        )


@pytest.mark.asyncio
async def test_direct_explicit_resolve_ip_never_invokes_doh(monkeypatch, no_fallback_env):
    from app.services import embeddings

    captured: dict = {}

    def fail_doh(*args, **kwargs):
        raise AssertionError("DoH must not run when a public resolve IP is explicit")

    def fake_post(url, payload, headers, timeout, resolve_ip):
        captured.update(url=url, resolve_ip=resolve_ip, headers=headers)
        return {"ok": True}

    monkeypatch.setattr(embeddings, "_resolve_public_provider_ip", fail_doh)
    monkeypatch.setattr(embeddings, "_post_json_with_pinned_resolve", fake_post)

    result = await embeddings.post_openai_compatible_json(
        "https://provider.example/v1/embeddings",
        {"model": "unit"},
        {"Authorization": "Bearer secret"},
        timeout=2.0,
        resolve_ip="8.8.8.8",
        purpose="embedding",
    )

    assert result == {"ok": True}
    assert captured["resolve_ip"] == "8.8.8.8"
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }


@pytest.mark.asyncio
async def test_local_bridge_requires_exact_route_and_stays_on_httpx(monkeypatch):
    from app.services import embeddings

    settings = SimpleNamespace(model_bridge_enabled=True, model_bridge_port=8765)
    calls: list[str] = []

    async def fake_local(url, payload, headers, timeout):
        calls.append(url)
        return {"ok": True}

    monkeypatch.setattr(embeddings, "get_settings", lambda: settings)
    monkeypatch.setattr(embeddings, "_post_json_to_local_bridge", fake_local)
    result = await embeddings.post_openai_compatible_json(
        "http://host.docker.internal:8765/chat/completions",
        {"model": "unit"},
        {"Authorization": "Bearer secret"},
        timeout=2.0,
        purpose="chat",
    )
    assert result == {"ok": True}
    assert calls == ["http://host.docker.internal:8765/chat/completions"]

    with pytest.raises(RuntimeError, match="chat transport must use"):
        await embeddings.post_openai_compatible_json(
            "http://host.docker.internal:8765/chat/completions?bypass=1",
            {"model": "unit"},
            {"Authorization": "Bearer secret"},
            timeout=2.0,
            purpose="chat",
        )
    assert len(calls) == 1

    direct_calls: list[tuple[str, str]] = []

    def fake_resolve(*_args, **_kwargs):
        return "8.8.8.8"

    def fake_direct(url, _payload, _headers, _timeout, resolve_ip):
        direct_calls.append((url, resolve_ip))
        return {"ok": "graph"}

    monkeypatch.setattr(embeddings, "_resolve_public_provider_ip", fake_resolve)
    monkeypatch.setattr(embeddings, "_post_json_with_pinned_resolve", fake_direct)

    with pytest.raises(RuntimeError, match="chat transport must use"):
        await embeddings.post_openai_compatible_json(
            "https://provider.example/v1/chat/completions",
            {"model": "unit"},
            {"Authorization": "Bearer secret"},
            timeout=2.0,
            purpose="chat",
        )
    assert direct_calls == []

    graph_result = await embeddings.post_openai_compatible_json(
        "https://graph.example/v1/chat/completions",
        {"model": "unit"},
        {"Authorization": "Bearer graph-secret"},
        timeout=2.0,
        purpose="graph",
    )
    assert graph_result == {"ok": "graph"}
    assert direct_calls == [
        ("https://graph.example:443/v1/chat/completions", "8.8.8.8")
    ]

    with pytest.raises(RuntimeError, match="Graph transport"):
        await embeddings.post_openai_compatible_json(
            "http://host.docker.internal:8765/chat/completions",
            {"model": "unit"},
            {"Authorization": "Bearer graph-secret"},
            timeout=2.0,
            purpose="graph",
        )

    with pytest.raises(ValueError, match="route does not match"):
        await embeddings.post_openai_compatible_json(
            "http://host.docker.internal:8765/chat/completions",
            {"model": "unit"},
            {"Authorization": "Bearer embedding-secret"},
            timeout=2.0,
            purpose="embedding",
        )

    with pytest.raises(ValueError, match="purpose is invalid"):
        await embeddings.post_openai_compatible_json(
            "https://provider.example/v1/chat/completions",
            {"model": "unit"},
            {"Authorization": "Bearer secret"},
            timeout=2.0,
            purpose="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_local_bridge_httpx_transport_disables_env_redirects_and_bounds_json(monkeypatch):
    from app.services import embeddings

    captured: dict = {}

    class FakeResponse:
        status_code = 200
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "identity",
            "Content-Length": "11",
        }

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_bytes(self):
            yield b'{"ok":true}'

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, *, content, headers):
            captured.update(
                method=method,
                url=url,
                content=content,
                headers=headers,
            )
            return FakeResponse()

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", FakeAsyncClient)

    result = await embeddings._post_json_to_local_bridge(
        "http://host.docker.internal:8765/embeddings",
        {"model": "unit"},
        {
            "Authorization": "Bearer secret",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
        2.0,
    )

    assert result == {"ok": True}
    assert captured["client_kwargs"] == {
        "timeout": 2.0,
        "trust_env": False,
        "follow_redirects": False,
    }
    assert captured["method"] == "POST"
    assert captured["url"] == "http://host.docker.internal:8765/embeddings"
    assert json.loads(captured["content"].decode()) == {"model": "unit"}


def test_pinned_connection_connects_to_ip_while_retaining_tls_hostname(monkeypatch):
    from app.services import embeddings

    captured: dict = {}
    sentinel_socket = object()

    def fake_create_connection(address, timeout, source_address):
        captured.update(
            address=address,
            timeout=timeout,
            source_address=source_address,
        )
        return sentinel_socket

    monkeypatch.setattr(embeddings.socket, "create_connection", fake_create_connection)
    connection = embeddings._PinnedHTTPSConnection(
        "provider.example",
        "8.8.8.8",
        443,
        timeout=3.0,
        context=ssl.create_default_context(),
    )

    created = connection._create_connection(
        ("provider.example", 443),
        3.0,
        None,
    )

    assert created is sentinel_socket
    assert connection.host == "provider.example"
    assert captured == {
        "address": ("8.8.8.8", 443),
        "timeout": 3.0,
        "source_address": None,
    }

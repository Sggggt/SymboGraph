from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_chat_answer_payload_has_no_output_token_cap(monkeypatch, no_fallback_env):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    captured: dict = {}

    async def fake_post_chat(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout
        captured["resolve_ip"] = resolve_ip
        return {"choices": [{"message": {"content": "Grounded answer [1]"}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_chat)

    result = await ChatProvider().answer_question_with_meta(
        "解释贝叶斯网络",
        [
            {
                "document_title": "Bayes",
                "partition": "Chapter 1",
                "content": "贝叶斯网络使用有向边表达条件依赖，并结合证据更新后验。",
            }
        ],
        [],
    )

    payload = captured["payload"]
    system_prompt = payload["messages"][0]["content"]
    assert result.answer == "Grounded answer [1]"
    assert payload["model"]
    assert "max_tokens" not in payload
    assert "complete as the supplied evidence supports" in system_prompt
    assert "concise" not in system_prompt.lower()


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

    async def fake_post_chat(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None) -> dict:
        calls.append({"url": url, "payload": payload, "headers": headers})
        return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_chat)

    await ChatProvider().classify_json("system", "user")
    await ChatProvider(purpose="graph").classify_json("system", "user")

    assert calls[0]["url"] == "https://chat.example.test/v1/chat/completions"
    assert calls[0]["payload"]["model"] == "chat-model"
    assert calls[0]["headers"]["Authorization"] == "Bearer chat-key"
    assert calls[1]["url"] == "https://graph.example.test/v1/chat/completions"
    assert calls[1]["payload"]["model"] == "graph-model"
    assert calls[1]["headers"]["Authorization"] == "Bearer graph-key"


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

    async def fake_post(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        captured["resolve_ip"] = resolve_ip
        return {"data": [{"embedding": [0.1] * 8}]}

    monkeypatch.setattr(runtime_settings, "sync_model_bridge_runtime_config", fake_sync_model_bridge_runtime_config)
    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post)

    result = await EmbeddingProvider().embed_texts_with_meta(["query"], text_type="query")

    assert result.external_called is True
    assert sync_calls == [model_bridge_client_base_url(8765)]
    assert captured["url"] == f"{model_bridge_client_base_url(8765)}/embeddings"
    assert captured["resolve_ip"] == "__none__"


@pytest.mark.asyncio
async def test_chat_provider_syncs_model_bridge_for_chat_only(monkeypatch, no_fallback_env):
    from app.core.config import get_settings, model_bridge_client_base_url
    from app.services import embeddings, runtime_settings
    from app.services.embeddings import ChatProvider

    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    get_settings.cache_clear()
    sync_calls: list[str] = []
    calls: list[str] = []

    def fake_sync_model_bridge_runtime_config(settings=None, env_entries=None, raise_on_error=True):
        sync_calls.append(settings.chat_base_url)
        return {"attempted": True, "ok": True, "config_version": "bridge-version"}

    async def fake_post(url: str, payload: dict, headers: dict, timeout: float, resolve_ip: str | None = None) -> dict:
        calls.append(url)
        return {"choices": [{"message": {"content": "{\"ok\": true}"}}]}

    monkeypatch.setattr(runtime_settings, "sync_model_bridge_runtime_config", fake_sync_model_bridge_runtime_config)
    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post)

    await ChatProvider().classify_json("system", "user")
    await ChatProvider(purpose="graph").classify_json("system", "user")

    assert sync_calls == [model_bridge_client_base_url(8765)]
    assert calls[0] == f"{model_bridge_client_base_url(8765)}/chat/completions"
    assert calls[1] == "https://graph.invalid/v1/chat/completions"


@pytest.mark.asyncio
async def test_openai_compatible_json_retries_transient_transport_errors(monkeypatch):
    from app.services import embeddings

    calls = 0
    sleeps: list[float] = []

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json, headers):
            nonlocal calls
            calls += 1
            if calls < embeddings.MODEL_REQUEST_MAX_ATTEMPTS:
                raise httpx.ConnectError("", request=httpx.Request("POST", url))
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"ok": True},
            )

    async def fake_sleep(seconds: float):
        sleeps.append(seconds)

    monkeypatch.setattr(embeddings.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(embeddings.asyncio, "sleep", fake_sleep)

    result = await embeddings.post_openai_compatible_json(
        "https://provider.example/v1/chat/completions",
        {"model": "unit"},
        {"Authorization": "Bearer redacted"},
        timeout=1,
    )

    assert result == {"ok": True}
    assert calls == embeddings.MODEL_REQUEST_MAX_ATTEMPTS
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]

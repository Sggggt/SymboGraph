from __future__ import annotations

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

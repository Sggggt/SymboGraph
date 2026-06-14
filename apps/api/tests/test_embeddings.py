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

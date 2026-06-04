from __future__ import annotations

import pytest


def write_test_env(monkeypatch, tmp_path, **entries: str):
    import app.core.config as config

    workspace = tmp_path / "workspace"
    app_dir = workspace / "apps" / "api" / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(config, "APP_DIR", app_dir)
    env_entries = {
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        "DATA_ROOT": (tmp_path / "data").as_posix(),
        "OPENAI_API_KEY": "unit-key",
        "EMBEDDING_BASE_URL": "https://embedding.test/v1",
        "EMBEDDING_API_KEY": "unit-embedding-key",
        **entries,
    }
    (workspace / ".env").write_text("\n".join(f"{key}={value}" for key, value in env_entries.items()) + "\n", encoding="utf-8")
    config.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_openai_compatible_embeddings_are_batched(no_fallback_env, monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import embeddings
    from app.services.embeddings import EmbeddingProvider

    write_test_env(monkeypatch, tmp_path, EMBEDDING_BATCH_SIZE="10", EMBEDDING_DIMENSIONS="2")
    get_settings.cache_clear()
    calls: list[list[str]] = []

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        batch = list(payload["input"])
        calls.append(batch)
        offset = sum(len(call) for call in calls[:-1])
        return {"data": [{"embedding": [float(offset + index + 1), 1.0]} for index, _text in enumerate(batch)]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await EmbeddingProvider().embed_texts_with_meta([f"text-{index}" for index in range(25)])

    assert [len(call) for call in calls] == [10, 10, 5]
    assert result.provider == "openai_compatible"
    assert result.external_called is True
    assert result.fallback_reason is None
    assert result.vectors == [[float(index + 1), 1.0] for index in range(25)]


@pytest.mark.asyncio
async def test_embedding_base_url_is_required_without_fallback(no_fallback_env, monkeypatch):
    from app.core.config import get_settings
    import app.core.config as config
    from app.services.embeddings import EmbeddingProvider, FallbackDisabledError

    monkeypatch.setattr(config, "WORKSPACE_ROOT", no_fallback_env.parent)
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "false")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    get_settings.cache_clear()

    with pytest.raises(FallbackDisabledError, match="EMBEDDING_BASE_URL"):
        await EmbeddingProvider().embed_texts_with_meta(["text"])


@pytest.mark.asyncio
async def test_openai_compatible_embeddings_reject_zero_vectors(no_fallback_env, monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import embeddings
    from app.services.embeddings import EmbeddingProvider

    write_test_env(monkeypatch, tmp_path, EMBEDDING_DIMENSIONS="2")
    get_settings.cache_clear()

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        return {"data": [{"embedding": [0.0, 0.0]}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    with pytest.raises(RuntimeError, match="all zeros"):
        await EmbeddingProvider().embed_texts_with_meta(["bad vector"])


@pytest.mark.asyncio
async def test_openai_compatible_embeddings_retry_without_dimensions(no_fallback_env, monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import embeddings
    from app.services.embeddings import EmbeddingProvider

    write_test_env(monkeypatch, tmp_path, EMBEDDING_DIMENSIONS="2")
    get_settings.cache_clear()
    payloads: list[dict] = []

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        payloads.append(dict(payload))
        if "dimensions" in payload:
            raise RuntimeError("InvalidParameter: dimensions is not supported")
        return {"data": [{"embedding": [1.0, 0.0]}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await EmbeddingProvider().embed_texts_with_meta(["text"])

    assert len(payloads) == 2
    assert payloads[0]["dimensions"] == 2
    assert "dimensions" not in payloads[1]
    assert result.vectors == [[1.0, 0.0]]


@pytest.mark.asyncio
async def test_openai_compatible_request_retries_transient_errors(no_fallback_env, monkeypatch):
    from app.services import embeddings

    calls = 0

    def flaky_curl(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("curl: (35) schannel: failed to receive handshake, SSL/TLS connection failed")
        return {"ok": True}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(embeddings, "_post_json_with_curl_resolve", flaky_curl)
    monkeypatch.setattr(embeddings.asyncio, "sleep", no_sleep)

    result = await embeddings.post_openai_compatible_json(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        {"model": "unit-test"},
        {"Authorization": "Bearer unit-test", "Content-Type": "application/json"},
        timeout=1,
        resolve_ip="127.0.0.1",
    )

    assert result == {"ok": True}
    assert calls == 3


@pytest.mark.asyncio
async def test_openai_compatible_request_retries_read_timeout(no_fallback_env, monkeypatch):
    import httpx

    from app.services import embeddings

    calls = 0

    def flaky_curl(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("")
        return {"ok": True}

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(embeddings, "_post_json_with_curl_resolve", flaky_curl)
    monkeypatch.setattr(embeddings.asyncio, "sleep", no_sleep)

    result = await embeddings.post_openai_compatible_json(
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        {"model": "unit-test"},
        {"Authorization": "Bearer unit-test", "Content-Type": "application/json"},
        timeout=1,
        resolve_ip="127.0.0.1",
    )

    assert result == {"ok": True}
    assert calls == 2


@pytest.mark.asyncio
async def test_openai_compatible_request_does_not_retry_invalid_parameters(no_fallback_env, monkeypatch):
    from app.services import embeddings

    calls = 0

    def invalid_parameter(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("<400> InternalError.Algo.InvalidParameter: batch size is invalid")

    monkeypatch.setattr(embeddings, "_post_json_with_curl_resolve", invalid_parameter)

    with pytest.raises(RuntimeError, match="InvalidParameter"):
        await embeddings.post_openai_compatible_json(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            {"model": "unit-test"},
            {"Authorization": "Bearer unit-test", "Content-Type": "application/json"},
            timeout=1,
            resolve_ip="127.0.0.1",
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_answer_prompt_enforces_latex_markdown_format(no_fallback_env, monkeypatch):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    captured_payload: dict = {}

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        captured_payload.update(payload)
        return {"choices": [{"message": {"content": "Use $$C_D(i) = \\frac{k_i}{n - 1}$$."}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await ChatProvider().answer_question_with_meta(
        "How is degree centrality defined?",
        [
            {
                "document_title": "Centrality Notes",
                "chapter": "L3",
                "content": "Degree centrality is ki over n minus 1.",
                "snippet": "Degree centrality is ki over n minus 1.",
                "citations": [],
                "metadata": {},
            }
        ],
        [],
    )

    system_prompt = captured_payload["messages"][0]["content"]
    user_prompt = captured_payload["messages"][-1]["content"]
    assert "valid LaTeX" in system_prompt
    assert "single dollar delimiters" in system_prompt
    assert "double dollar delimiters" in system_prompt
    assert "Never write formulas as glued plain text" in system_prompt
    assert "\\frac{k_i}{n - 1}" in system_prompt
    assert "variables are not attached to neighboring words" in user_prompt
    assert result.answer.startswith("Use $$")


@pytest.mark.asyncio
async def test_answer_prompt_language_follows_english_question_with_chinese_context(no_fallback_env, monkeypatch):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    captured_payload: dict = {}

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        captured_payload.update(payload)
        return {"choices": [{"message": {"content": "A graph is represented by vertices and edges."}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await ChatProvider().answer_question_with_meta(
        "What is a graph and how is it represented?",
        [
            {
                "document_title": "图论讲义",
                "chapter": "第1章",
                "content": "图由顶点集合和边集合组成。",
                "snippet": "图由顶点集合和边集合组成。",
                "citations": [],
                "metadata": {},
            }
        ],
        [],
    )

    system_prompt = captured_payload["messages"][0]["content"]
    user_prompt = captured_payload["messages"][-1]["content"]
    assert "Required answer language: English" in system_prompt
    assert "Required answer language: English" in user_prompt
    assert "do not switch the answer language to match the excerpt language" in system_prompt
    assert result.answer.startswith("A graph")


@pytest.mark.asyncio
async def test_answer_prompt_language_follows_chinese_question_with_english_context(no_fallback_env, monkeypatch):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    captured_payload: dict = {}

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        captured_payload.update(payload)
        return {"choices": [{"message": {"content": "图由顶点集合和边集合组成。"}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await ChatProvider().answer_question_with_meta(
        "\u4ec0\u4e48\u662f\u56fe\uff0c\u56fe\u5982\u4f55\u8868\u793a\uff1f",
        [
            {
                "document_title": "Graph Notes",
                "chapter": "Chapter 1",
                "content": "A graph consists of a vertex set and an edge set.",
                "snippet": "A graph consists of a vertex set and an edge set.",
                "citations": [],
                "metadata": {},
            }
        ],
        [],
    )

    system_prompt = captured_payload["messages"][0]["content"]
    user_prompt = captured_payload["messages"][-1]["content"]
    assert "Required answer language: Chinese" in system_prompt
    assert "Required answer language: Chinese" in user_prompt
    assert "do not switch the answer language to match the excerpt language" in system_prompt
    assert result.answer.startswith("\u56fe")


def test_no_context_answer_follows_question_language(no_fallback_env):
    from app.services.embeddings import no_context_answer

    assert no_context_answer("What is a graph?").startswith("I could not find")
    assert no_context_answer("\u4ec0\u4e48\u662f\u56fe\uff1f").startswith("\u8bfe\u7a0b\u6750\u6599")


@pytest.mark.asyncio
async def test_chat_json_response_format_falls_back_to_prompt_only(no_fallback_env, monkeypatch):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    response_formats: list[object] = []

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        response_formats.append(payload.get("response_format"))
        if payload.get("response_format"):
            raise RuntimeError("InvalidParameter: response_format is not supported")
        return {"choices": [{"message": {"content": '{"route":"retrieve_notes"}'}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await ChatProvider().classify_json("Return JSON.", "Classify.", fallback={})

    assert response_formats == [{"type": "json_object"}, None]
    assert result == {"route": "retrieve_notes"}


@pytest.mark.asyncio
async def test_graph_json_schema_falls_back_to_json_object_then_prompt_only(no_fallback_env, monkeypatch):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    response_formats: list[object] = []

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        response_formats.append(payload.get("response_format"))
        if payload.get("response_format"):
            raise RuntimeError("InvalidParameter: response_format is not supported")
        return {"choices": [{"message": {"content": '{"concepts":[],"relations":[]}'}}]}

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await ChatProvider().extract_graph_payload("Graph material", "L1", "pdf")

    assert [item.get("type") if isinstance(item, dict) else None for item in response_formats] == ["json_schema", "json_object", None]
    assert result == {"concepts": [], "relations": []}


@pytest.mark.asyncio
async def test_chat_content_parts_are_normalized(no_fallback_env, monkeypatch):
    from app.services import embeddings
    from app.services.embeddings import ChatProvider

    async def fake_post_json(url, payload, headers, *, timeout, resolve_ip=None):
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Part one "},
                            {"type": "text", "text": {"value": "part two"}},
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(embeddings, "post_openai_compatible_json", fake_post_json)

    result = await ChatProvider().answer_question_with_meta(
        "Question",
        [
            {
                "document_title": "Doc",
                "chapter": "L1",
                "content": "Evidence",
                "snippet": "Evidence",
                "metadata": {},
            }
        ],
        [],
    )

    assert result.answer == "Part one part two"


@pytest.mark.asyncio
async def test_classify_json_without_fallback_raises_on_exception(no_fallback_env, monkeypatch):
    """Regression: classify_json without fallback must propagate exceptions instead of silently returning empty dict."""
    from app.services.embeddings import ChatProvider

    async def failing_post(*args, **kwargs):
        raise RuntimeError("model error")

    monkeypatch.setattr(ChatProvider, "_post_chat_json_with_response_format_fallback", failing_post)

    with pytest.raises(RuntimeError, match="model error"):
        await ChatProvider().classify_json("system", "user")

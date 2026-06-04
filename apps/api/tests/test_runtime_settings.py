
from __future__ import annotations


def _redirect_settings_env(monkeypatch, workspace):
    from app.core import config
    from app.services import runtime_settings

    app_dir = workspace / "apps" / "api" / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(config, "APP_DIR", app_dir)
    monkeypatch.setattr(runtime_settings, "ENV_PATH", workspace / ".env")
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", workspace / ".env.example")
    config.get_settings.cache_clear()


def test_env_sync_detects_bom_key(tmp_path, monkeypatch):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text("\ufeffDATABASE_URL=sqlite:///test.db\nRERANKER_ENABLED=true\n", encoding="utf-8")
    example_path.write_text("DATABASE_URL=\nRERANKER_ENABLED=false\n", encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)

    before = runtime_settings.env_sync_status()
    assert before["bom_keys"] == ["DATABASE_URL"]
    assert before["missing_keys"] == []

    runtime_settings.normalize_env_file()
    after = runtime_settings.env_sync_status()
    assert after["bom_keys"] == []
    assert "\ufeffDATABASE_URL" not in env_path.read_text(encoding="utf-8")


def test_env_sync_detects_key_mismatch(tmp_path, monkeypatch):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text("DATABASE_URL=sqlite:///test.db\nEXTRA_ONLY=true\n", encoding="utf-8")
    example_path.write_text("DATABASE_URL=\nOPENAI_API_KEY=\n", encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)

    status = runtime_settings.env_sync_status()

    assert status["synced"] is False
    assert status["missing_keys"] == ["OPENAI_API_KEY"]
    assert status["extra_keys"] == ["EXTRA_ONLY"]


def test_runtime_check_skips_reranker_when_disabled(tmp_path, monkeypatch):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    env_path.write_text("RERANKER_ENABLED=false\n", encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)

    class Settings:
        reranker_enabled = False
        reranker_device = "cpu"
        reranker_model = "BAAI/bge-reranker-v2-m3"
        reranker_url = "http://reranker:8080/rerank"
        qdrant_url = "http://qdrant:6333"
        redis_url = "redis://redis:6379/0"
        chat_base_url = "https://api.openai.com/v1"

    monkeypatch.setattr(runtime_settings, "get_settings", lambda: Settings())
    monkeypatch.setattr(runtime_settings, "env_sync_status", lambda: {"synced": True, "missing_keys": [], "extra_keys": [], "bom_keys": []})
    monkeypatch.setattr(runtime_settings, "_check_postgres", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_qdrant", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_redis", lambda: True)

    def fail_http(*args, **kwargs):
        raise AssertionError("reranker health should not be called")

    monkeypatch.setattr(runtime_settings.httpx, "get", fail_http)

    payload = runtime_settings.runtime_check_payload()
    assert payload["reranker"]["enabled"] is False
    assert payload["blocking_issues"] == []


def test_runtime_check_reports_reranker_status(tmp_path, monkeypatch):
    from app.services import runtime_settings
    from app.services import reranker

    env_path = tmp_path / ".env"
    env_path.write_text("RERANKER_ENABLED=true\nRERANKER_MODEL=unit-test-reranker\n", encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(reranker, "_reranker_instance", None)
    monkeypatch.setattr(reranker, "_reranker_error", None)

    class Settings:
        reranker_enabled = True
        reranker_model = "BAAI/bge-reranker-v2-m3"
        reranker_max_length = 512
        qdrant_url = "http://qdrant:6333"
        redis_url = "redis://redis:6379/0"
        chat_base_url = "https://api.openai.com/v1"

    monkeypatch.setattr(runtime_settings, "get_settings", lambda: Settings())
    monkeypatch.setattr(runtime_settings, "env_sync_status", lambda: {"synced": True, "missing_keys": [], "extra_keys": [], "bom_keys": []})
    monkeypatch.setattr(runtime_settings, "_check_postgres", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_qdrant", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_redis", lambda: True)

    def mock_load_reranker():
        raise RuntimeError("model not available")

    monkeypatch.setattr(reranker, "_load_reranker", mock_load_reranker)

    payload = runtime_settings.runtime_check_payload()
    # When reranker_enabled=True but model fails to load, enabled stays True
    # (config says enable it, but runtime cannot load it)
    assert payload["reranker"]["enabled"] is True
    assert payload["reranker"]["reachable"] is False
    assert payload["reranker"]["healthy"] is False


def test_runtime_check_reports_model_bridge_when_configured(monkeypatch):
    from app.services import runtime_settings

    class Settings:
        reranker_enabled = False
        reranker_device = "cpu"
        reranker_model = "BAAI/bge-reranker-v2-m3"
        reranker_url = "http://reranker:8080/rerank"
        qdrant_url = "http://qdrant:6333"
        redis_url = "redis://redis:6379/0"
        chat_base_url = "http://host.docker.internal:8765"

    class Response:
        status_code = 200

    monkeypatch.setattr(runtime_settings, "get_settings", lambda: Settings())
    monkeypatch.setattr(runtime_settings, "env_sync_status", lambda: {"synced": True, "missing_keys": [], "extra_keys": [], "bom_keys": []})
    monkeypatch.setattr(runtime_settings, "_check_postgres", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_qdrant", lambda: True)
    monkeypatch.setattr(runtime_settings, "_check_redis", lambda: True)
    monkeypatch.setattr(runtime_settings.httpx, "get", lambda *args, **kwargs: Response())

    payload = runtime_settings.runtime_check_payload()

    assert payload["infrastructure"]["model_bridge"] is True
    assert payload["warnings"] == []


def test_settings_routes_compose_model_calls_through_bridge(monkeypatch, tmp_path):
    from app.core.config import get_settings

    _redirect_settings_env(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                f"DATABASE_URL=sqlite:///{(tmp_path / 'test.db').as_posix()}",
                f"DATA_ROOT={(tmp_path / 'data').as_posix()}",
                "MODEL_BRIDGE_ENABLED=true",
                "MODEL_BRIDGE_PORT=8766",
                "CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1",
                "CHAT_RESOLVE_IP=1.2.3.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("API_CHAT_BASE_URL", raising=False)
    monkeypatch.delenv("API_CHAT_RESOLVE_IP", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.chat_base_url == "http://host.docker.internal:8766"
    assert settings.chat_resolve_ip == "__none__"
    assert settings.model_bridge_enabled is True

    get_settings.cache_clear()


def test_model_settings_payload_uses_split_model_urls(monkeypatch, tmp_path):
    from app.core.config import get_settings
    from app.services import runtime_settings

    _redirect_settings_env(monkeypatch, tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=unit-chat-key",
                "CHAT_BASE_URL=https://chat.example/v1",
                "CHAT_RESOLVE_IP=1.2.3.4",
                "EMBEDDING_API_KEY=unit-embedding-key",
                "EMBEDDING_BASE_URL=https://embedding.example/v1",
                "EMBEDDING_RESOLVE_IP=5.6.7.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "unit-chat-key")
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "false")
    monkeypatch.delenv("API_CHAT_BASE_URL", raising=False)
    monkeypatch.delenv("API_CHAT_RESOLVE_IP", raising=False)
    monkeypatch.setenv("CHAT_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("CHAT_RESOLVE_IP", "1.2.3.4")
    monkeypatch.setenv("EMBEDDING_API_KEY", "unit-embedding-key")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "https://embedding.example/v1")
    monkeypatch.setenv("EMBEDDING_RESOLVE_IP", "5.6.7.8")
    get_settings.cache_clear()

    payload = runtime_settings.model_settings_payload()

    assert payload["chat_base_url"] == "https://chat.example/v1"
    assert payload["embedding_base_url"] == "https://embedding.example/v1"
    assert payload["chat_resolve_ip"] == "1.2.3.4"
    assert payload["embedding_resolve_ip"] == "5.6.7.8"
    assert payload["degraded_mode"] is False

    get_settings.cache_clear()


def test_update_model_settings_updates_current_process_env(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    _redirect_settings_env(monkeypatch, tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("EMBEDDING_MODEL=text-embedding-v4\n", encoding="utf-8")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    get_settings.cache_clear()

    payload = runtime_settings.update_model_settings({"embedding_model": "text-embedding-v3"})

    assert payload["embedding_model"] == "text-embedding-v3"
    assert env_path.read_text(encoding="utf-8").strip() == "EMBEDDING_MODEL=text-embedding-v3"
    assert get_settings().embedding_model == "text-embedding-v3"


def test_update_model_settings_updates_graph_extraction_concurrency(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    _redirect_settings_env(monkeypatch, tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("GRAPH_EXTRACTION_CONCURRENCY=2\n", encoding="utf-8")
    monkeypatch.setenv("GRAPH_EXTRACTION_CONCURRENCY", "2")
    get_settings.cache_clear()

    payload = runtime_settings.update_model_settings({"graph_extraction_concurrency": 4})

    assert payload["graph_extraction_concurrency"] == 4
    assert env_path.read_text(encoding="utf-8").strip() == "GRAPH_EXTRACTION_CONCURRENCY=4"
    assert get_settings().graph_extraction_concurrency == 4


def test_update_model_settings_updates_model_request_timeout(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    _redirect_settings_env(monkeypatch, tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("MODEL_REQUEST_TIMEOUT_SECONDS=180\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_REQUEST_TIMEOUT_SECONDS", "180")
    get_settings.cache_clear()

    payload = runtime_settings.update_model_settings({"model_request_timeout_seconds": 45})

    assert payload["model_request_timeout_seconds"] == 45
    assert env_path.read_text(encoding="utf-8").strip() == "MODEL_REQUEST_TIMEOUT_SECONDS=45"
    assert get_settings().model_request_timeout_seconds == 45


def test_update_model_settings_updates_graph_community_summary_toggle(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.services import runtime_settings

    _redirect_settings_env(monkeypatch, tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("ENABLE_GRAPH_COMMUNITY_SUMMARIES=true\n", encoding="utf-8")
    monkeypatch.setenv("ENABLE_GRAPH_COMMUNITY_SUMMARIES", "true")
    get_settings.cache_clear()

    payload = runtime_settings.update_model_settings({"enable_graph_community_summaries": False})

    assert payload["enable_graph_community_summaries"] is False
    assert env_path.read_text(encoding="utf-8").strip() == "ENABLE_GRAPH_COMMUNITY_SUMMARIES=false"
    assert get_settings().enable_graph_community_summaries is False


def test_settings_hot_reload_prefers_workspace_env_over_process_env(tmp_path, monkeypatch):
    from app.core import config

    workspace = tmp_path / "workspace"
    _redirect_settings_env(monkeypatch, workspace)
    env_path = workspace / ".env"
    env_path.write_text(
        "\n".join(
            [
                f"DATABASE_URL=sqlite:///{(tmp_path / 'test.db').as_posix()}",
                f"DATA_ROOT={(tmp_path / 'data').as_posix()}",
                "CHAT_MODEL=qwen3.7-plus",
                "GRAPH_EXTRACTION_CONCURRENCY=3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHAT_MODEL", "deepseek-v4-pro")
    config.get_settings.cache_clear()

    assert config.get_settings().chat_model == "qwen3.7-plus"
    env_path.write_text(
        "\n".join(
            [
                f"DATABASE_URL=sqlite:///{(tmp_path / 'test.db').as_posix()}",
                f"DATA_ROOT={(tmp_path / 'data').as_posix()}",
                "CHAT_MODEL=qwen3.7-plus-hot",
                "GRAPH_EXTRACTION_CONCURRENCY=4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reloaded = config.get_settings()
    assert reloaded.chat_model == "qwen3.7-plus-hot"
    assert reloaded.graph_extraction_concurrency == 4


def test_update_model_settings_does_not_remove_keys_when_cleared(tmp_path, monkeypatch):
    """清除 API Key 或 resolve_ip 时，不应从 .env 中删除 key，否则会导致与 .env.example 不一致。"""
    from app.core.config import get_settings
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_text(
        "OPENAI_API_KEY=secret\nCHAT_RESOLVE_IP=1.2.3.4\nEMBEDDING_API_KEY=secret2\nEMBEDDING_RESOLVE_IP=5.6.7.8\n",
        encoding="utf-8",
    )
    example_path.write_text(
        "OPENAI_API_KEY=\nCHAT_BASE_URL=\nCHAT_RESOLVE_IP=\nEMBEDDING_API_KEY=\nEMBEDDING_BASE_URL=\nEMBEDDING_RESOLVE_IP=\nEMBEDDING_MODEL=\nCHAT_MODEL=\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("CHAT_RESOLVE_IP", "1.2.3.4")
    monkeypatch.setenv("EMBEDDING_API_KEY", "secret2")
    monkeypatch.setenv("EMBEDDING_RESOLVE_IP", "5.6.7.8")
    get_settings.cache_clear()

    # Simulate clearing both API keys and resolve IPs
    payload = runtime_settings.update_model_settings(
        {
            "chat_base_url": "https://chat.example/v1",
            "embedding_base_url": "https://embedding.example/v1",
            "chat_resolve_ip": None,
            "embedding_resolve_ip": None,
            "clear_api_key": True,
            "clear_embedding_api_key": True,
            "embedding_model": "text-embedding-v3",
            "chat_model": "gpt-4",
        }
    )

    # Keys should still be present in .env (with empty values)
    env_text = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in env_text
    assert "EMBEDDING_API_KEY=" in env_text
    assert "CHAT_RESOLVE_IP=" in env_text
    assert "EMBEDDING_RESOLVE_IP=" in env_text

    # env_sync should still be consistent
    sync = runtime_settings.env_sync_status()
    assert sync["synced"] is True
    assert sync["missing_keys"] == []
    assert sync["extra_keys"] == []

    get_settings.cache_clear()

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_bridge_status_treats_explicit_default_https_port_as_same_identity(
    monkeypatch,
):
    from app.services import runtime_settings

    settings = SimpleNamespace(
        model_bridge_enabled=True,
        model_bridge_port=8765,
        model_bridge_admin_token="unit-bridge-token",
        model_request_timeout_seconds=180,
        chat_base_url="http://host.docker.internal:8765",
        chat_api_protocol="anthropic",
        embedding_api_protocol="openai",
    )
    env_entries = {
        "CHAT_API_PROTOCOL": "anthropic",
        "CHAT_BASE_URL": "https://chat.example.test/api",
        "CHAT_RESOLVE_IP": "1.1.1.1",
        "EMBEDDING_API_PROTOCOL": "openai",
        "EMBEDDING_BASE_URL": "https://embedding.example.test/v1",
        "EMBEDDING_RESOLVE_IP": "2.2.2.2",
    }

    class Response:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return dict(self._payload)

    def fake_request(_method, path, **_kwargs):
        common = {
            "config_version": "unit-version",
            "chat_target_hash": "chat-hash",
            "embedding_target_hash": "embedding-hash",
            "chat_api_protocol": "anthropic",
            "embedding_api_protocol": "openai",
            "timeout": 180,
        }
        if path == "/health":
            return Response({**common, "routes": {"/v1/messages": "chat_anthropic"}})
        return Response(
            {
                **common,
                "chat_target_base_url": "https://chat.example.test/api",
                "chat_resolve_ip": "1.1.1.1",
                "embedding_target_base_url": "https://embedding.example.test/v1",
                "embedding_resolve_ip": "2.2.2.2",
            }
        )

    monkeypatch.setattr(runtime_settings, "_bridge_http_request", fake_request)

    payload = runtime_settings.model_bridge_status_payload(
        settings=settings,
        env_entries=env_entries,
    )

    assert payload["reachable"] is True
    assert payload["admin_available"] is True
    assert payload["config_matches"] is True


@pytest.mark.parametrize(
    "process_token",
    [
        None,
        "",
        "   ",
        "\tunit-token",
        "unit-token\n",
        "unit-token\x01",
        "default",
        "local-model-bridge-admin",
        '\"\"',
        '\"   \"',
        '\"default\"',
        '\"unit-token\x01\"',
        "''",
        "'   '",
        "'default'",
        "'unit-token\x01'",
    ],
)
def test_active_update_rejects_absent_or_invalid_process_bridge_token_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    no_fallback_env,
    process_token: str | None,
):
    from app.core.config import get_settings
    from app.services import runtime_settings

    env_path = tmp_path / "managed-runtime.env"
    original = "\n".join(
        [
            "MODEL_BRIDGE_ENABLED=true",
            "MODEL_BRIDGE_PORT=8765",
            "MODEL_BRIDGE_ADMIN_TOKEN=stale-valid-managed-token",
            "CHAT_BASE_URL=https://chat.example.test/v1",
            "CHAT_RESOLVE_IP=1.1.1.1",
            "EMBEDDING_BASE_URL=https://embedding.example.test/v1",
            "EMBEDDING_RESOLVE_IP=2.2.2.2",
        ]
    ) + "\n"
    env_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    if process_token is None:
        monkeypatch.delenv("MODEL_BRIDGE_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", process_token)
    get_settings.cache_clear()

    effects: list[str] = []
    for name in (
        "normalize_env_file",
        "_update_env_file",
        "_apply_runtime_env",
        "publish_runtime_settings_version",
        "reload_model_bridge",
    ):
        monkeypatch.setattr(
            runtime_settings,
            name,
            lambda *args, _name=name, **kwargs: effects.append(_name),
        )

    settings = get_settings()
    next_cadence = 2 if settings.gray_zone_observation_cadence != 2 else 3
    with pytest.raises(ValueError) as exc_info:
        runtime_settings.update_model_settings(
            {
                "chat_base_url": "https://chat.example.test/v2",
                "gray_zone_observation_cadence": next_cadence,
            }
        )

    assert "MODEL_BRIDGE_ADMIN_TOKEN" in str(exc_info.value)
    assert "stale-valid-managed-token" not in str(exc_info.value)
    assert effects == []
    assert env_path.read_text(encoding="utf-8") == original
    get_settings.cache_clear()


def test_workspace_env_parser_decodes_managed_writer_quotes(monkeypatch, tmp_path):
    from app.core import config

    env_path = tmp_path / "managed-runtime.env"
    env_path.write_text(
        "\n".join(
            [
                'MODEL_BRIDGE_ADMIN_TOKEN="default"',
                "SINGLE_QUOTED_TOKEN='default'",
                'PADDED_VALUE="   "',
                'ESCAPED_VALUE="quoted\\\"value"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "_active_runtime_env_path", lambda: env_path)

    entries = config._read_workspace_env()

    assert entries["MODEL_BRIDGE_ADMIN_TOKEN"] == "default"
    assert entries["SINGLE_QUOTED_TOKEN"] == "default"
    assert entries["PADDED_VALUE"] == "   "
    assert entries["ESCAPED_VALUE"] == 'quoted"value'


def test_runtime_env_normalization_retains_root_only_service_keys():
    from app.services import runtime_settings

    original = (
        b"MODEL_BRIDGE_ADMIN_TOKEN=stale-managed-secret\n"
        b"SEMANTIC_CHUNKING_ENABLED=true\n"
        b"CHAT_MODEL=unit-model\n"
    )

    normalized = runtime_settings._normalized_runtime_env_bytes(
        original,
        remove_deprecated=False,
    )

    assert b"MODEL_BRIDGE_ADMIN_TOKEN=stale-managed-secret" in normalized
    assert b"SEMANTIC_CHUNKING_ENABLED=true" in normalized
    assert b"CHAT_MODEL=unit-model" in normalized


def test_runtime_env_update_preserves_service_start_key_in_single_root_file(monkeypatch, tmp_path):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    original = (
        b"MODEL_BRIDGE_ADMIN_TOKEN=stale-managed-secret\n"
        b"CHAT_MODEL=old-model\n"
    )
    published: dict[str, bytes] = {}
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(
        runtime_settings,
        "runtime_env_file_lock",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        runtime_settings,
        "_require_runtime_env_recovery_resolved",
        lambda _path: None,
    )
    monkeypatch.setattr(
        runtime_settings,
        "_read_runtime_env_snapshot",
        lambda _path: ({"identity_hash": "before"}, original),
    )
    monkeypatch.setattr(
        runtime_settings,
        "_cleanup_runtime_env_temporary_files",
        lambda _path: None,
    )
    monkeypatch.setattr(
        runtime_settings,
        "_atomic_replace_runtime_env_bytes",
        lambda content, **_kwargs: published.update(content=content),
    )

    runtime_settings._update_env_file({"CHAT_MODEL": "new-model"})

    assert published["content"] == (
        b"MODEL_BRIDGE_ADMIN_TOKEN=stale-managed-secret\n"
        b"CHAT_MODEL=new-model\n"
    )


def test_runtime_env_update_allows_service_start_key_in_the_single_root_file(monkeypatch, tmp_path):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    env_path.write_text("CHAT_MODEL=unit-model\n", encoding="utf-8")
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)

    runtime_settings._update_env_file(
        {"MODEL_BRIDGE_ADMIN_TOKEN": "unit-test-secret"}
    )

    assert "MODEL_BRIDGE_ADMIN_TOKEN=unit-test-secret" in env_path.read_text(encoding="utf-8")


def test_chat_protocol_reload_failure_restores_exact_env_and_bridge_before_publish(
    monkeypatch,
    tmp_path,
    no_fallback_env,
):
    from app.core.config import get_settings
    from app.services import runtime_settings

    env_path = tmp_path / "managed-runtime.env"
    original = (
        "MODEL_BRIDGE_ENABLED=true\n"
        "MODEL_BRIDGE_PORT=8765\n"
        "CHAT_API_PROTOCOL=openai\n"
        "CHAT_BASE_URL=https://chat.example.test/v1\n"
        "CHAT_RESOLVE_IP=1.1.1.1\n"
        "EMBEDDING_API_PROTOCOL=openai\n"
        "EMBEDDING_BASE_URL=https://embedding.example.test/v1\n"
        "EMBEDDING_RESOLVE_IP=2.2.2.2\n"
    ).encode("utf-8")
    env_path.write_bytes(original)
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setenv("MODEL_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("MODEL_BRIDGE_PORT", "8765")
    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", "unit-bridge-token")
    monkeypatch.setenv("CHAT_API_PROTOCOL", "openai")
    monkeypatch.setenv("CHAT_BASE_URL", "http://host.docker.internal:8765")
    get_settings.cache_clear()
    monkeypatch.setattr(
        runtime_settings,
        "preflight_model_bridge_reload",
        lambda **_kwargs: {
            "desired": {},
            "previous": {
                "chat_api_protocol": "openai",
                "chat_target_base_url": "https://chat.example.test:443/v1",
                "chat_resolve_ip": "1.1.1.1",
                "embedding_api_protocol": "openai",
                "embedding_target_base_url": "https://embedding.example.test:443/v1",
                "embedding_resolve_ip": "2.2.2.2",
                "timeout": 180,
            },
        },
    )
    reload_calls: list[dict[str, str]] = []

    def fake_reload(*, settings, env_entries):
        reload_calls.append(dict(env_entries))
        if len(reload_calls) == 1:
            return {"attempted": True, "ok": False, "status_code": 503}
        return {"attempted": True, "ok": True, "config_version": "restored"}

    monkeypatch.setattr(runtime_settings, "reload_model_bridge", fake_reload)
    monkeypatch.setattr(
        runtime_settings,
        "publish_runtime_settings_version",
        lambda **_kwargs: pytest.fail("rejected reload must not publish a version"),
    )

    with pytest.raises(ValueError, match="exact env bytes.*restored"):
        runtime_settings.update_model_settings(
            {
                "chat_api_protocol": "anthropic",
                "chat_base_url": "https://chat.example.test",
            }
        )

    assert env_path.read_bytes() == original
    assert len(reload_calls) == 2
    assert reload_calls[0]["CHAT_API_PROTOCOL"] == "anthropic"
    assert reload_calls[1]["CHAT_API_PROTOCOL"] == "openai"
    assert reload_calls[1]["EMBEDDING_API_PROTOCOL"] == "openai"
    assert get_settings().chat_api_protocol == "openai"
    get_settings.cache_clear()


def test_bridge_admin_http_never_uses_proxy_environment(monkeypatch):
    from app.services import runtime_settings

    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def request(self, method, url, **kwargs):
            captured.update(method=method, url=url, request_kwargs=kwargs)
            return SimpleNamespace(status_code=200)

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setattr(runtime_settings.httpx, "Client", FakeClient)
    settings = SimpleNamespace(
        chat_base_url="http://host.docker.internal:8765",
        model_bridge_port=8765,
    )

    response = runtime_settings._bridge_http_request(
        "GET",
        "/admin/config",
        settings=settings,
        headers={"X-Bridge-Admin-Token": "unit-process-token"},
        timeout=3.0,
    )

    assert response.status_code == 200
    assert captured["client_kwargs"] == {
        "trust_env": False,
        "follow_redirects": False,
    }
    assert captured["url"] == "http://host.docker.internal:8765/admin/config"


def test_runtime_env_apply_rejects_process_only_key_without_overwrite(monkeypatch):
    from app.services import runtime_settings

    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", "process-only-unit-secret")

    with pytest.raises(ValueError, match="Process-only environment keys"):
        runtime_settings._apply_runtime_env(
            {"MODEL_BRIDGE_ADMIN_TOKEN": "managed-secret-must-not-apply"}
        )

    assert (
        runtime_settings.os.environ["MODEL_BRIDGE_ADMIN_TOKEN"]
        == "process-only-unit-secret"
    )


def test_managed_runtime_refresh_preserves_process_only_token(monkeypatch):
    from app.services import runtime_settings

    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", "process-only-unit-secret")
    monkeypatch.setattr(
        runtime_settings,
        "_RUNTIME_ENV_PROCESS_APPLIED_VALUES",
        {"MODEL_BRIDGE_ADMIN_TOKEN": "stale-managed-secret"},
    )
    monkeypatch.setattr(
        runtime_settings,
        "_env_entries",
        lambda _path: {
            "MODEL_BRIDGE_ADMIN_TOKEN": "stale-managed-secret",
            "CHAT_MODEL": "managed-unit-model",
        },
    )

    runtime_settings._apply_runtime_env_file_to_process_environment()

    assert (
        runtime_settings.os.environ["MODEL_BRIDGE_ADMIN_TOKEN"]
        == "process-only-unit-secret"
    )
    assert runtime_settings.os.environ["CHAT_MODEL"] == "managed-unit-model"
    assert (
        "MODEL_BRIDGE_ADMIN_TOKEN"
        not in runtime_settings._RUNTIME_ENV_PROCESS_APPLIED_VALUES
    )


def test_single_root_env_retains_service_start_secret_without_hot_applying_it(
    monkeypatch,
    tmp_path,
):
    from app.services import runtime_settings

    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    env_path.write_bytes(
        b"MODEL_BRIDGE_ADMIN_TOKEN=stale-managed-secret\nCHAT_MODEL=unit-model\n"
    )
    example_path.write_bytes(
        b"MODEL_BRIDGE_ADMIN_TOKEN=\nCHAT_MODEL=unit-model\n"
    )
    monkeypatch.setattr(runtime_settings, "ENV_PATH", env_path)
    monkeypatch.setattr(runtime_settings, "ENV_EXAMPLE_PATH", example_path)

    initial_status = runtime_settings.env_sync_status()

    assert initial_status["synced"] is True
    assert initial_status["missing_keys"] == []
    assert initial_status["extra_keys"] == []
    assert initial_status["deprecated_keys"] == []

    env_path.write_bytes(
        runtime_settings._normalized_runtime_env_bytes(
            env_path.read_bytes(),
            remove_deprecated=False,
        )
    )
    normalized_status = runtime_settings.env_sync_status()

    assert normalized_status["synced"] is True
    assert "MODEL_BRIDGE_ADMIN_TOKEN=stale-managed-secret" in env_path.read_text(encoding="utf-8")

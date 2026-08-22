from __future__ import annotations

import importlib.util
import http.client
import gzip
import io
import inspect
import json
import ssl
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def load_model_bridge_module():
    module_path = Path(__file__).resolve().parents[3] / "infra" / "model-bridge" / "model_bridge.py"
    spec = importlib.util.spec_from_file_location("model_bridge_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_embedding_protocol_is_openai_only_and_binds_bridge_config_version():
    bridge = load_model_bridge_module()
    config = bridge.build_config(
        chat_api_protocol="anthropic",
        embedding_api_protocol="openai",
        chat_target_base_url="https://chat.example.test",
        chat_resolve_ip="1.1.1.1",
        embedding_target_base_url="https://embedding.example.test/v1",
        embedding_resolve_ip="2.2.2.2",
        timeout=30,
    )
    public = bridge.public_config(config)
    assert config.embedding_api_protocol == "openai"
    assert public["embedding_api_protocol"] == "openai"
    assert config.config_version == bridge.stable_hash(
        bridge.config_version_payload(
            chat_api_protocol="anthropic",
            embedding_api_protocol="openai",
            chat_target_base_url="https://chat.example.test",
            chat_resolve_ip="1.1.1.1",
            embedding_target_base_url="https://embedding.example.test/v1",
            embedding_resolve_ip="2.2.2.2",
            timeout=30,
        )
    )
    with pytest.raises(ValueError, match="embedding_api_protocol must be openai"):
        bridge.build_config(
            chat_target_base_url="https://chat.example.test/v1",
            chat_resolve_ip="1.1.1.1",
            embedding_api_protocol="anthropic",
            embedding_target_base_url="https://embedding.example.test/v1",
            embedding_resolve_ip="2.2.2.2",
            timeout=30,
        )


def read_json(url: str, *, token: str | None = None) -> dict:
    headers = {"X-Bridge-Admin-Token": token} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict, *, token: str) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "X-Bridge-Admin-Token": token},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_verified_https_opener_does_not_inject_default_user_agent():
    module = load_model_bridge_module()
    opener = module.build_verified_https_opener(module.verified_tls_context())

    assert opener.addheaders == []
    https_handlers = [
        handler
        for handler in opener.handlers
        if isinstance(handler, urllib.request.HTTPSHandler)
    ]
    assert len(https_handlers) == 1
    request = urllib.request.Request(
        "https://provider.example.test:8443/v1/messages",
        data=b"{}",
        method="POST",
        headers={
            "X-Api-Key": "unit-only-secret",
            "Anthropic-Version": module.ANTHROPIC_VERSION,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
        },
    )

    processed = https_handlers[0].https_request(request)
    headers = {
        key.casefold(): value for key, value in processed.header_items()
    }

    assert headers == {
        "x-api-key": "unit-only-secret",
        "anthropic-version": module.ANTHROPIC_VERSION,
        "content-type": "application/json",
        "accept": "application/json",
        "accept-encoding": "identity",
        "content-length": "2",
        "host": "provider.example.test:8443",
    }
    assert "user-agent" not in headers
    assert "cookie" not in headers


def test_model_bridge_routes_embeddings_to_embedding_target():
    module = load_model_bridge_module()

    assert module.route_for_path("/embeddings") == "embedding"
    assert module.route_for_path("/chat/completions") == "chat"
    assert module.route_for_path("/v1/messages") == "chat_anthropic"
    assert module.route_for_path("/v1/embeddings") is None
    assert module.route_for_path("/embeddings?encoding=float") is None
    assert module.route_for_path("/embeddings?") is None
    assert module.route_for_path("/embeddings#") is None
    assert module.route_for_path("/chat/completions/") is None
    assert module.route_for_path("/unknown") is None


@pytest.mark.parametrize(
    "request_target",
    [
        "http://attacker.example/chat/completions",
        "https://attacker.example/embeddings",
        "//attacker.example/chat/completions",
        "//attacker.example/embeddings",
        "chat/completions",
        "embeddings",
        "scheme:/chat/completions",
    ],
)
def test_model_bridge_rejects_non_origin_form_request_targets(request_target):
    module = load_model_bridge_module()

    assert module.route_for_path(request_target) is None


def test_model_bridge_handler_does_not_forward_absolute_form_request_target(
    monkeypatch,
):
    module = load_model_bridge_module()
    config = module.build_config(
        chat_target_base_url="https://chat.example.test/v1",
        chat_resolve_ip="1.1.1.1",
        embedding_target_base_url="https://embedding.example.test/v1",
        embedding_resolve_ip="2.2.2.2",
        timeout=30,
    )
    module.BridgeState.configure(config, "unit-token")
    forwarded = False

    def fail_if_forwarded(*_args, **_kwargs):
        nonlocal forwarded
        forwarded = True
        raise AssertionError("absolute-form request target must not be forwarded")

    monkeypatch.setattr(
        module.ModelBridgeHandler,
        "_forward_with_urllib",
        fail_if_forwarded,
    )
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.ModelBridgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        server.server_address[1],
        timeout=5,
    )
    try:
        connection.putrequest(
            "POST",
            "http://attacker.example/chat/completions",
        )
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Content-Length", "2")
        connection.endheaders(b"{}")
        response = connection.getresponse()

        assert response.status == 404
        response.read()
        assert forwarded is False
    finally:
        connection.close()
        server.shutdown()
        server.server_close()


def test_windows_launcher_does_not_reintroduce_credential_bearing_curl_fallback():
    launcher = (
        Path(__file__).resolve().parents[3] / "start-app.ps1"
    ).read_text(encoding="utf-8")

    assert "model_bridge.py" in launcher
    assert "curl.exe" not in launcher
    assert "requires Windows curl" not in launcher
    assert '"--admin-token"' not in launcher
    assert 'GetEnvironmentVariable(\n  "MODEL_BRIDGE_ADMIN_TOKEN"' not in launcher
    assert 'Get-DotEnvValue -Key "MODEL_BRIDGE_ADMIN_TOKEN"' in launcher
    assert "generated ephemerally for this launcher process" not in launcher
    assert "requires a non-default MODEL_BRIDGE_ADMIN_TOKEN" in launcher
    assert "$processModelBridgeAdminToken" not in launcher
    assert "$modelBridgeAdminToken -ne $modelBridgeAdminToken.Trim()" in launcher
    assert "[char]::IsControl($character)" in launcher
    assert "$modelBridgeAdminToken.Trim().ToLowerInvariant()" not in launcher

    compose = (
        Path(__file__).resolve().parents[3] / "infra" / "docker-compose.yml"
    ).read_text(encoding="utf-8")
    assert compose.count(
        "MODEL_BRIDGE_ADMIN_TOKEN: ${MODEL_BRIDGE_ADMIN_TOKEN:-}"
    ) == 4
    assert 'profiles: ["model-bridge"]' in compose
    assert "container_name: course-kg-model-bridge" in compose
    assert "--allow-private-container-bind" in compose
    assert "MODEL_BRIDGE_CLIENT_HOST: model-bridge" in compose
    assert '$${CHAT_BASE_URL:-}' in compose
    assert '$${EMBEDDING_BASE_URL:-}' in compose
    model_bridge_service = compose.split("  model-bridge:", 1)[1].split(
        "\n  api:", 1
    )[0]
    assert "ports:" not in model_bridge_service
    assert "expose:" in model_bridge_service
    parsed_compose = yaml.safe_load(compose)
    bridge_command = parsed_compose["services"]["model-bridge"]["command"]
    assert isinstance(bridge_command, list)
    assert len(bridge_command) == 1
    assert bridge_command[0].startswith("exec /app/apps/api/.venv/bin/python")


def test_model_bridge_client_host_is_closed_and_compose_only(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config, "running_in_container", lambda: True)
    monkeypatch.setenv("MODEL_BRIDGE_CLIENT_HOST", "model-bridge")
    assert config.model_bridge_client_base_url(8765) == "http://model-bridge:8765"

    monkeypatch.setenv("MODEL_BRIDGE_CLIENT_HOST", "attacker.example")
    with pytest.raises(ValueError, match="allowlisted local bridge host"):
        config.model_bridge_client_base_url(8765)


def test_model_bridge_requires_ca_and_hostname_verification():
    module = load_model_bridge_module()

    context = module.verified_tls_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    source = inspect.getsource(module)
    assert "ssl.CERT_NONE" not in source
    assert "check_hostname = False" not in source
    assert "--insecure" not in source
    assert '"insecure\\n"' not in source
    assert 'add_argument("--admin-token"' not in source


@pytest.mark.parametrize(
    ("target", "resolve_ip"),
    [
        ("http://provider.example.test/v1", "1.1.1.1"),
        ("https://user:secret@provider.example.test/v1", "1.1.1.1"),
        ("https://provider.example.test/v1?route=unsafe", "1.1.1.1"),
        ("https://provider.example.test/v1?", "1.1.1.1"),
        ("https://provider.example.test/v1#", "1.1.1.1"),
        ("https://provider.example.test/v1\x7f", "1.1.1.1"),
        ("https://127.0.0.1/v1", "1.1.1.1"),
        ("https://10.0.0.1/v1", "1.1.1.1"),
        ("https://127.1/v1", "1.1.1.1"),
        ("https://2130706433/v1", "1.1.1.1"),
        ("https://0x7f000001/v1", "1.1.1.1"),
        ("https://localhost/v1", "1.1.1.1"),
        ("https://localhost./v1", "1.1.1.1"),
        ("https://sub.localhost/v1", "1.1.1.1"),
        ("https://provider.example.test/v1", "127.0.0.1"),
        ("https://provider.example.test/v1", "10.0.0.1"),
        ("https://provider.example.test/v1", "100.64.0.1"),
        ("https://provider.example.test/v1", "224.0.0.1"),
        ("https://provider.example.test/v1", "ff02::1"),
        ("https://provider.example.test/v1", "::ffff:127.0.0.1"),
        ("https://provider.example.test/v1", "fec0::1"),
    ],
)
def test_model_bridge_rejects_insecure_targets_and_private_resolve_ips(
    target,
    resolve_ip,
):
    module = load_model_bridge_module()

    with pytest.raises(ValueError):
        module.build_config(
            chat_target_base_url=target,
            chat_resolve_ip=resolve_ip,
            embedding_target_base_url="https://embedding.example.test/v1",
            embedding_resolve_ip="2.2.2.2",
            timeout=30,
        )


def test_model_bridge_urllib_forwards_only_allowlisted_headers(
    monkeypatch,
):
    module = load_model_bridge_module()
    handler = object.__new__(module.ModelBridgeHandler)
    handler.headers = {
        "Authorization": "Bearer unit-only-secret",
        "Content-Type": "application/json",
        "Cookie": "must-not-forward",
        "X-Bridge-Admin-Token": "must-not-forward",
        "X-Custom-Secret": "must-not-forward",
    }
    captured: dict[str, str] = {}
    captured_handlers: list[object] = []

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json", "Content-Encoding": "identity"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            return b'{"ok":true}'

    class FakeOpener:
        def open(self, request, timeout):
            del timeout
            captured.update(
                {key.casefold(): value for key, value in request.header_items()}
            )
            return FakeResponse()

    def fake_build_opener(*handlers):
        captured_handlers.extend(handlers)
        return FakeOpener()

    monkeypatch.setattr(module.urllib.request, "build_opener", fake_build_opener)

    status_code, response_body = handler._forward_with_urllib(
        "https://provider.example.test/v1/chat/completions",
        b'{"messages":[]}',
        route="chat",
        timeout=5,
        resolve_ip="1.1.1.1",
    )

    assert status_code == 200
    assert response_body == b'{"ok":true}'
    assert captured == {
        "authorization": "Bearer unit-only-secret",
        "content-type": "application/json",
        "accept": "application/json",
        "accept-encoding": "identity",
        "host": "provider.example.test",
    }
    assert any(
        isinstance(handler, module.RejectRedirectHandler)
        for handler in captured_handlers
    )


def test_model_bridge_anthropic_route_is_protocol_bound_and_header_isolated(
    monkeypatch,
):
    module = load_model_bridge_module()
    config = module.build_config(
        chat_api_protocol="anthropic",
        chat_target_base_url="https://provider.example.test",
        chat_resolve_ip="1.1.1.1",
        embedding_target_base_url="https://embedding.example.test/v1",
        embedding_resolve_ip="2.2.2.2",
        timeout=30,
    )
    assert module.route_matches_protocol(config, "chat_anthropic") is True
    assert module.route_matches_protocol(config, "chat") is False
    public = module.public_config(config)
    assert public["chat_api_protocol"] == "anthropic"
    assert public["routes"] == {
        "/v1/messages": "chat_anthropic",
        "/embeddings": "embedding",
    }

    handler = object.__new__(module.ModelBridgeHandler)
    handler.headers = {
        "Authorization": "Bearer unit-only-secret",
        "Anthropic-Version": module.ANTHROPIC_VERSION,
        "Cookie": "must-not-forward",
    }
    captured: dict[str, str] = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/json", "Content-Encoding": "identity"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            return b'{"content":[{"type":"text","text":"ok"}]}'

    class FakeOpener:
        def open(self, request, timeout):
            del timeout
            captured.update(
                {key.casefold(): value for key, value in request.header_items()}
            )
            return FakeResponse()

    monkeypatch.setattr(
        module,
        "build_verified_https_opener",
        lambda _context: FakeOpener(),
    )
    status_code, _response = handler._forward_with_urllib(
        "https://provider.example.test/v1/messages",
        b'{"messages":[]}',
        route="chat_anthropic",
        timeout=5,
        resolve_ip="1.1.1.1",
    )

    assert status_code == 200
    assert captured == {
        "authorization": "Bearer unit-only-secret",
        "anthropic-version": module.ANTHROPIC_VERSION,
        "content-type": "application/json",
        "accept": "application/json",
        "accept-encoding": "identity",
        "host": "provider.example.test",
    }
    assert "x-api-key" not in captured


@pytest.mark.parametrize(
    ("target_url", "resolve_ip", "expected_host"),
    [
        (
            "https://provider.example.test:8443/v1/chat/completions",
            "1.1.1.1",
            "provider.example.test:8443",
        ),
        (
            "https://[2001:4860:4860::8888]:8443/v1/chat/completions",
            "2606:4700:4700::1111",
            "[2001:4860:4860::8888]:8443",
        ),
    ],
)
def test_model_bridge_preserves_exact_host_authority(
    monkeypatch,
    target_url,
    resolve_ip,
    expected_host,
):
    module = load_model_bridge_module()
    handler = object.__new__(module.ModelBridgeHandler)
    handler.headers = {"Authorization": "Bearer unit-only-secret"}
    captured: dict[str, str] = {}

    class FakeResponse:
        status = 200
        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "identity",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            return b'{"ok":true}'

    class FakeOpener:
        def open(self, request, timeout):
            del timeout
            captured.update(
                {key.casefold(): value for key, value in request.header_items()}
            )
            return FakeResponse()

    monkeypatch.setattr(
        module,
        "build_verified_https_opener",
        lambda _context: FakeOpener(),
    )

    status_code, _response = handler._forward_with_urllib(
        target_url,
        b'{"messages":[]}',
        route="chat",
        timeout=5,
        resolve_ip=resolve_ip,
    )

    assert status_code == 200
    assert captured["host"] == expected_host


def test_model_bridge_rejects_cross_protocol_auth_before_body_or_upstream(
    monkeypatch,
):
    module = load_model_bridge_module()
    token = "unit-admin-token"
    module.BridgeState.configure(
        module.build_config(
            chat_api_protocol="anthropic",
            chat_target_base_url="https://provider.example.test",
            chat_resolve_ip="1.1.1.1",
            embedding_target_base_url="https://embedding.example.test/v1",
            embedding_resolve_ip="2.2.2.2",
            timeout=30,
        ),
        token,
    )
    upstream_calls = 0

    def fail_forward(*_args, **_kwargs):
        nonlocal upstream_calls
        upstream_calls += 1
        raise AssertionError("invalid auth must not reach the provider")

    monkeypatch.setattr(
        module.ModelBridgeHandler,
        "_forward_with_urllib",
        fail_forward,
    )
    server = module.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        module.ModelBridgeHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/v1/messages",
            data=b"not-json-and-must-not-be-read",
            method="POST",
            headers={
                "X-Api-Key": "must-not-cross-protocol",
                "Anthropic-Version": module.ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(request, timeout=5)
        assert captured.value.code == 401
        assert upstream_calls == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    ("protocol", "chat_base", "embedding_base"),
    [
        ("anthropic", "https://provider.example.test/v1", "https://embedding.example.test/v1"),
        ("anthropic", "https://provider.example.test/v1/messages", "https://embedding.example.test/v1"),
        ("openai", "https://provider.example.test/v1/chat/completions", "https://embedding.example.test/v1"),
        ("openai", "https://provider.example.test/v1", "https://embedding.example.test/v1/embeddings"),
    ],
)
def test_model_bridge_rejects_base_urls_that_would_duplicate_fixed_routes(
    protocol,
    chat_base,
    embedding_base,
):
    module = load_model_bridge_module()

    with pytest.raises(ValueError, match="fixed request route|must not end in /v1"):
        module.build_config(
            chat_api_protocol=protocol,
            chat_target_base_url=chat_base,
            chat_resolve_ip="1.1.1.1",
            embedding_target_base_url=embedding_base,
            embedding_resolve_ip="2.2.2.2",
            timeout=30,
        )


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_model_bridge_rejects_redirect_before_creating_followup_request(
    status_code,
):
    module = load_model_bridge_module()
    handler = module.RejectRedirectHandler()
    request = urllib.request.Request(
        "https://provider.example.test/v1/chat/completions",
        data=b"{}",
        method="POST",
        headers={"Authorization": "Bearer unit-only-secret"},
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        handler.redirect_request(
            request,
            None,
            status_code,
            "Found",
            {"Location": "http://127.0.0.1/private"},
            "http://127.0.0.1/private",
        )

    assert error.value.code == status_code
    assert error.value.url == request.full_url


def test_model_bridge_doh_pins_allowlisted_resolver_ip_without_system_dns(
    monkeypatch,
):
    module = load_model_bridge_module()
    captured_handlers: list[object] = []
    read_limits: list[int] = []
    requests: list[urllib.request.Request] = []
    connected_addresses: list[object] = []
    system_dns_calls: list[object] = []

    monkeypatch.setattr(
        module,
        "DOH_RESOLVERS",
        (("resolver.example.test", "/dns-query", ("1.1.1.1",)),),
    )

    def forbidden_system_dns(*args, **kwargs):
        system_dns_calls.append((args, kwargs))
        raise AssertionError("DoH bootstrap must not use system DNS")

    monkeypatch.setattr(module, "_original_getaddrinfo", forbidden_system_dns)

    class FakeResponse:
        status = 200
        headers = {
            "Content-Type": "application/dns-json; charset=utf-8",
            "Content-Encoding": "identity",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit):
            read_limits.append(limit)
            return (
                b'{"Status":0,"TC":false,'
                b'"Answer":[{"type":1,"data":"8.8.8.8"}]}'
            )

    class FakeOpener:
        def open(self, request, timeout):
            assert timeout == 5
            requests.append(request)
            parsed = module.urlparse(request.full_url)
            connected_addresses.extend(
                module._thread_safe_getaddrinfo(parsed.hostname, 443)
            )
            return FakeResponse()

    def fake_build_opener(*handlers):
        captured_handlers.extend(handlers)
        return FakeOpener()

    monkeypatch.setattr(module.urllib.request, "build_opener", fake_build_opener)

    assert module.resolve_public_a_record("provider.example.test") == "8.8.8.8"
    assert read_limits == [module.MAX_DOH_RESPONSE_BODY_BYTES + 1]
    assert system_dns_calls == []
    assert requests[0].full_url.startswith(
        "https://resolver.example.test/dns-query?"
    )
    assert {key.casefold(): value for key, value in requests[0].header_items()} == {
        "accept": "application/dns-json",
        "accept-encoding": "identity",
    }
    assert connected_addresses == [
        (module.socket.AF_INET, module.socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))
    ]
    assert getattr(module._thread_local, "dns_overrides", None) is None
    assert getattr(module._thread_local, "forbid_system_dns", False) is False
    assert any(
        isinstance(handler, module.RejectRedirectHandler)
        for handler in captured_handlers
    )
    https_handlers = [
        handler
        for handler in captured_handlers
        if isinstance(handler, module.urllib.request.HTTPSHandler)
    ]
    assert len(https_handlers) == 1
    assert https_handlers[0]._context.verify_mode == ssl.CERT_REQUIRED
    assert https_handlers[0]._context.check_hostname is True


def test_model_bridge_doh_forbids_unpinned_hostname_during_bootstrap(
    monkeypatch,
):
    module = load_model_bridge_module()
    system_dns_calls: list[object] = []
    monkeypatch.setattr(
        module,
        "DOH_RESOLVERS",
        (("resolver.example.test", "/dns-query", ("1.1.1.1",)),),
    )

    def forbidden_system_dns(*args, **kwargs):
        system_dns_calls.append((args, kwargs))
        raise AssertionError("system DNS fallback was attempted")

    monkeypatch.setattr(module, "_original_getaddrinfo", forbidden_system_dns)

    class FakeOpener:
        def open(self, _request, timeout):
            assert timeout == 5
            module._thread_safe_getaddrinfo("unexpected.example.test", 443)
            raise AssertionError("unreachable")

    monkeypatch.setattr(
        module,
        "build_verified_https_opener",
        lambda _context: FakeOpener(),
    )

    assert module.resolve_public_a_record("provider.example.test") is None
    assert system_dns_calls == []
    assert getattr(module._thread_local, "dns_overrides", None) is None
    assert getattr(module._thread_local, "forbid_system_dns", False) is False


@pytest.mark.parametrize(
    ("status", "content_type", "content_encoding"),
    [
        (204, "application/dns-json", "identity"),
        (200, "text/plain", "identity"),
        (200, "application/dns-json;", "identity"),
        (200, "application/dns-json; charset=ascii", "identity"),
        (200, "application/dns-json; charset=utf-8; charset=utf-8", "identity"),
        (200, "application/dns-json", "gzip"),
    ],
)
def test_model_bridge_doh_rejects_invalid_http_envelope(
    monkeypatch,
    status,
    content_type,
    content_encoding,
):
    module = load_model_bridge_module()
    monkeypatch.setattr(
        module,
        "DOH_RESOLVERS",
        (("resolver.example.test", "/dns-query", ("1.1.1.1",)),),
    )

    class FakeResponse:
        headers = {
            "Content-Type": content_type,
            "Content-Encoding": content_encoding,
        }

        def __init__(self):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return (
                b'{"Status":0,"TC":false,'
                b'"Answer":[{"type":1,"data":"8.8.8.8"}]}'
            )

    class FakeOpener:
        def open(self, _request, timeout):
            assert timeout == 5
            return FakeResponse()

    monkeypatch.setattr(
        module,
        "build_verified_https_opener",
        lambda _context: FakeOpener(),
    )

    assert module.resolve_public_a_record("provider.example.test") is None


@pytest.mark.parametrize(
    "content_type",
    [
        "application/dns-json",
        "application/dns-json; charset=utf-8",
        "application/json",
        "application/json; charset=utf-8",
    ],
)
def test_model_bridge_doh_accepts_strict_json_media_types(
    monkeypatch,
    content_type,
):
    module = load_model_bridge_module()
    monkeypatch.setattr(
        module,
        "DOH_RESOLVERS",
        (("resolver.example.test", "/dns-query", ("1.1.1.1",)),),
    )

    class FakeResponse:
        status = 200
        headers = {"Content-Type": content_type}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit):
            return (
                b'{"Status":0,"TC":false,"Answer":['
                b'{"type":5,"data":"alias.example.test."},'
                b'{"type":28,"data":"2606:4700:4700::1111"},'
                b'{"type":1,"data":"8.8.8.8"}]}'
            )

    class FakeOpener:
        def open(self, _request, timeout):
            assert timeout == 5
            return FakeResponse()

    monkeypatch.setattr(
        module,
        "build_verified_https_opener",
        lambda _context: FakeOpener(),
    )

    assert module.resolve_public_a_record("provider.example.test") == "8.8.8.8"


@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b'{"Status":NaN,"TC":false,"Answer":[]}',
        b'{"Status":0,"Status":0,"TC":false,"Answer":[]}',
        b'{"Status":2,"TC":false,"Answer":[]}',
        b'{"Status":0,"Answer":[]}',
        b'{"Status":0,"TC":true,"Answer":[]}',
        b'{"Status":0,"TC":false,"Answer":{}}',
        b'{"Status":0,"TC":false,"Answer":[null]}',
        b'{"Status":0,"TC":false,"Answer":[{"type":"1","data":"8.8.8.8"}]}',
        b'{"Status":0,"TC":false,"Answer":[{"type":0,"data":"8.8.8.8"}]}',
        b'{"Status":0,"TC":false,"Answer":[{"type":65536,"data":"8.8.8.8"}]}',
        b'{"Status":0,"TC":false,"Answer":[{"type":1,"data":"127.0.0.1"}]}',
        b'{"Status":0,"TC":false,"Answer":[{"type":1,"data":"100.64.0.1"}]}',
        b'{"Status":0,"TC":false,"Answer":[{"type":1,"data":" 8.8.8.8"}]}',
    ],
)
def test_model_bridge_doh_dns_payload_is_strict_and_fail_closed(body):
    module = load_model_bridge_module()

    with pytest.raises(ValueError):
        module._validated_doh_public_a_response(body)


def test_model_bridge_doh_rejects_oversized_answer_count_and_body(monkeypatch):
    module = load_model_bridge_module()
    monkeypatch.setattr(module, "MAX_DOH_ANSWER_RECORDS", 1)
    two_answers = (
        b'{"Status":0,"TC":false,"Answer":['
        b'{"type":1,"data":"8.8.8.8"},'
        b'{"type":1,"data":"1.1.1.1"}]}'
    )
    with pytest.raises(ValueError, match="answer count"):
        module._validated_doh_public_a_response(two_answers)

    monkeypatch.setattr(module, "MAX_DOH_RESPONSE_BODY_BYTES", 8)
    with pytest.raises(ValueError, match="byte bound"):
        module._validated_doh_public_a_response(b"123456789")


def test_model_bridge_doh_skips_valid_non_a_records_and_requires_public_a():
    module = load_model_bridge_module()
    only_non_a = (
        b'{"Status":0,"TC":false,"Answer":['
        b'{"type":5,"data":"alias.example.test."},'
        b'{"type":28,"data":"2606:4700:4700::1111"}]}'
    )
    assert module._validated_doh_public_a_response(only_non_a) is None

    mixed = (
        b'{"Status":0,"TC":false,"Answer":['
        b'{"type":5,"data":"alias.example.test."},'
        b'{"type":1,"data":"8.8.8.8"}]}'
    )
    assert module._validated_doh_public_a_response(mixed) == "8.8.8.8"


def test_model_bridge_dns_resolution_failure_never_falls_back_to_system_dns(
    monkeypatch,
):
    module = load_model_bridge_module()
    config = module.build_config(
        chat_target_base_url="https://provider.example.test/v1",
        chat_resolve_ip=None,
        embedding_target_base_url="https://embedding.example.test/v1",
        embedding_resolve_ip=None,
        timeout=30,
    )
    module.BridgeState.configure(config, "unit-token")
    resolution_attempts: list[str] = []

    def no_public_answer(hostname: str):
        resolution_attempts.append(hostname)
        return None

    monkeypatch.setattr(module, "resolve_public_a_record", no_public_answer)

    with pytest.raises(RuntimeError, match="no verified public A record"):
        module.BridgeState.resolve_target_ip(
            "chat",
            "provider.example.test",
            None,
        )

    assert resolution_attempts == ["provider.example.test"]


@pytest.mark.parametrize(
    "token",
    [
        "",
        " ",
        "\tunit-token",
        "unit-token\n",
        "local-model-bridge-admin",
        "change-me",
        "default",
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
def test_model_bridge_rejects_missing_or_unsafe_admin_token(token):
    module = load_model_bridge_module()
    config = module.build_config(
        chat_target_base_url="https://provider.example.test/v1",
        chat_resolve_ip="1.1.1.1",
        embedding_target_base_url="https://embedding.example.test/v1",
        embedding_resolve_ip="2.2.2.2",
        timeout=30,
    )

    with pytest.raises(ValueError, match="MODEL_BRIDGE_ADMIN_TOKEN"):
        module.BridgeState.configure(config, token)


@pytest.mark.parametrize(
    "admin_token",
    [
        None,
        "",
        " ",
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
def test_model_bridge_main_requires_valid_admin_token_before_binding(
    monkeypatch,
    admin_token,
):
    module = load_model_bridge_module()
    if admin_token is None:
        monkeypatch.delenv("MODEL_BRIDGE_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", admin_token)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            host="127.0.0.1",
            port=8765,
            target_base_url="",
            resolve_ip="",
            chat_target_base_url="https://provider.example.test/v1",
            embedding_target_base_url="https://embedding.example.test/v1",
            chat_resolve_ip="1.1.1.1",
            embedding_resolve_ip="2.2.2.2",
            timeout=30,
        ),
    )
    server_bound = False

    def fail_if_server_is_created(*_args, **_kwargs):
        nonlocal server_bound
        server_bound = True
        raise AssertionError("server must not bind without an admin token")

    monkeypatch.setattr(module, "ThreadingHTTPServer", fail_if_server_is_created)

    with pytest.raises(ValueError, match="MODEL_BRIDGE_ADMIN_TOKEN"):
        module.main()

    assert server_bound is False


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "localhost", "example.test"])
def test_model_bridge_main_rejects_nonliteral_or_nonloopback_bind_before_server(
    monkeypatch,
    host,
):
    module = load_model_bridge_module()
    monkeypatch.setenv("MODEL_BRIDGE_ADMIN_TOKEN", "unit-token")
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            host=host,
            port=8765,
            target_base_url="",
            resolve_ip="",
            chat_target_base_url="https://provider.example.test/v1",
            embedding_target_base_url="https://embedding.example.test/v1",
            chat_resolve_ip="1.1.1.1",
            embedding_resolve_ip="2.2.2.2",
            timeout=30,
        ),
    )
    server_bound = False

    def fail_if_server_is_created(*_args, **_kwargs):
        nonlocal server_bound
        server_bound = True
        raise AssertionError("server must bind only to a literal loopback")

    monkeypatch.setattr(module, "ThreadingHTTPServer", fail_if_server_is_created)

    with pytest.raises(ValueError, match="literal loopback"):
        module.main()

    assert server_bound is False


def test_model_bridge_allows_explicit_private_container_bind(monkeypatch):
    module = load_model_bridge_module()
    monkeypatch.setattr(module, "running_in_container", lambda: True)

    assert module.verified_bind_host(
        "0.0.0.0",
        allow_private_container_bind=True,
    ) == "0.0.0.0"


def test_model_bridge_container_detection_is_boolean():
    module = load_model_bridge_module()

    assert isinstance(module.running_in_container(), bool)


def test_model_bridge_rejects_private_container_bind_outside_container(monkeypatch):
    module = load_model_bridge_module()
    monkeypatch.setattr(module, "running_in_container", lambda: False)

    with pytest.raises(ValueError, match="literal loopback"):
        module.verified_bind_host(
            "0.0.0.0",
            allow_private_container_bind=True,
        )


def test_model_bridge_provider_http_error_discards_raw_response(monkeypatch):
    module = load_model_bridge_module()
    handler = object.__new__(module.ModelBridgeHandler)
    handler.headers = {"Authorization": "Bearer unit-only-secret"}
    sentinel = b"SENTINEL_RAW_PROVIDER_ERROR_RESPONSE"

    class FakeOpener:
        def open(self, request, timeout):
            del request, timeout
            raise urllib.error.HTTPError(
                "https://provider.example.test/v1/chat/completions",
                400,
                "provider error",
                {},
                io.BytesIO(sentinel),
            )

    monkeypatch.setattr(
        module.urllib.request,
        "build_opener",
        lambda *_args: FakeOpener(),
    )

    status_code, response_body = handler._forward_with_urllib(
        "https://provider.example.test/v1/chat/completions",
        b'{"messages":[]}',
        route="chat",
        timeout=5,
        resolve_ip="1.1.1.1",
    )

    assert status_code == 400
    assert sentinel not in response_body
    assert json.loads(response_body) == {
        "error": {"code": "upstream_http_error", "status": 400}
    }


def test_model_bridge_provider_redirect_becomes_fixed_502_without_location(
    monkeypatch,
):
    module = load_model_bridge_module()
    handler = object.__new__(module.ModelBridgeHandler)
    handler.headers = {"Authorization": "Bearer unit-only-secret"}
    sentinel = "SENTINEL_REDIRECT_LOCATION"

    class FakeOpener:
        def open(self, request, timeout):
            del request, timeout
            raise urllib.error.HTTPError(
                "https://provider.example.test/v1/chat/completions",
                302,
                "Found",
                {"Location": f"http://127.0.0.1/{sentinel}"},
                io.BytesIO(sentinel.encode("utf-8")),
            )

    monkeypatch.setattr(
        module.urllib.request,
        "build_opener",
        lambda *_args: FakeOpener(),
    )

    status_code, response_body = handler._forward_with_urllib(
        "https://provider.example.test/v1/chat/completions",
        b'{"messages":[]}',
        route="chat",
        timeout=5,
        resolve_ip="1.1.1.1",
    )

    assert status_code == 502
    assert sentinel.encode("utf-8") not in response_body
    assert json.loads(response_body) == {
        "error": {"code": "upstream_redirect_rejected", "status": 302}
    }


def test_model_bridge_gzip_response_decompression_is_bounded(monkeypatch):
    module = load_model_bridge_module()
    monkeypatch.setattr(module, "MAX_MODEL_RESPONSE_BODY_BYTES", 1024)
    valid = gzip.compress(b'{"ok":true}')
    assert module.validated_json_response_body(valid) == b'{"ok":true}'

    bomb = gzip.compress(b'{"data":"' + (b"x" * 2048) + b'"}')
    with pytest.raises(RuntimeError, match="decompression failed"):
        module.validated_json_response_body(bomb)


@pytest.mark.parametrize(
    ("content_type", "content_encoding"),
    [
        ("text/html", "identity"),
        ("application/json", "gzip"),
    ],
)
def test_model_bridge_rejects_invalid_provider_headers_before_body_read(
    monkeypatch,
    content_type,
    content_encoding,
):
    module = load_model_bridge_module()
    handler = object.__new__(module.ModelBridgeHandler)
    handler.headers = {"Authorization": "Bearer unit-only-secret"}
    read_called = False

    class FakeResponse:
        status = 200
        headers = {
            "Content-Type": content_type,
            "Content-Encoding": content_encoding,
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, *_args):
            nonlocal read_called
            read_called = True
            return b"SENTINEL_PROVIDER_BODY"

    class FakeOpener:
        def open(self, _request, timeout):
            del timeout
            return FakeResponse()

    monkeypatch.setattr(
        module,
        "build_verified_https_opener",
        lambda _context: FakeOpener(),
    )

    with pytest.raises(RuntimeError, match="Verified upstream HTTPS forwarding failed"):
        handler._forward_with_urllib(
            "https://provider.example.test/v1/chat/completions",
            b'{"messages":[]}',
            route="chat",
            timeout=5,
            resolve_ip="1.1.1.1",
        )
    assert read_called is False


@pytest.mark.parametrize(
    "raw",
    [
        b'{"ok":true,"ok":false}',
        b'{"value":NaN}',
    ],
)
def test_model_bridge_rejects_non_strict_provider_json(raw):
    module = load_model_bridge_module()

    with pytest.raises(RuntimeError, match="strict JSON"):
        module.validated_json_response_body(raw)


def test_model_bridge_unknown_routes_and_errors_fail_closed_without_sentinels(
    monkeypatch,
    capsys,
):
    module = load_model_bridge_module()
    token = "unit-token"
    config = module.build_config(
        chat_target_base_url="https://chat.example.test/v1",
        chat_resolve_ip="__none__",
        embedding_target_base_url="https://embedding.example.test/v1",
        embedding_resolve_ip="__none__",
        timeout=30,
    )
    module.BridgeState.configure(config, token)
    sentinel = "SENTINEL_AUTH_BODY_PROVIDER_RESPONSE"

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        module.ModelBridgeHandler,
        "_forward_with_urllib",
        fail_forward,
    )
    server = module.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        module.ModelBridgeHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        unknown = urllib.request.Request(
            f"{base_url}/v1/embeddings?leak={sentinel}",
            data=b"{}",
            method="POST",
            headers={
                "Authorization": f"Bearer {sentinel}",
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as unknown_error:
            urllib.request.urlopen(unknown, timeout=5)
        assert unknown_error.value.code == 404

        unsupported = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=b"not-json",
            method="POST",
            headers={
                "Authorization": f"Bearer {sentinel}",
                "Content-Type": "text/plain",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as unsupported_error:
            urllib.request.urlopen(unsupported, timeout=5)
        assert unsupported_error.value.code == 415

        invalid_json = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=b"not-json",
            method="POST",
            headers={
                "Authorization": f"Bearer {sentinel}",
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as invalid_json_error:
            urllib.request.urlopen(invalid_json, timeout=5)
        assert invalid_json_error.value.code == 400

        oversized_connection = http.client.HTTPConnection(
            "127.0.0.1",
            server.server_address[1],
            timeout=5,
        )
        oversized_connection.putrequest("POST", "/embeddings")
        oversized_connection.putheader(
            "Content-Length",
            str(module.MAX_MODEL_REQUEST_BODY_BYTES + 1),
        )
        oversized_connection.putheader(
            "Authorization",
            f"Bearer {sentinel}",
        )
        oversized_connection.putheader("Content-Type", "application/json")
        oversized_connection.endheaders()
        oversized_response = oversized_connection.getresponse()
        assert oversized_response.status == 413
        oversized_response.read()
        oversized_connection.close()

        admin_oversized = http.client.HTTPConnection(
            "127.0.0.1",
            server.server_address[1],
            timeout=5,
        )
        admin_oversized.putrequest("POST", "/admin/reload")
        admin_oversized.putheader(
            "X-Bridge-Admin-Token",
            token,
        )
        admin_oversized.putheader(
            "Content-Length",
            str(module.MAX_ADMIN_RELOAD_BODY_BYTES + 1),
        )
        admin_oversized.putheader("Content-Type", "application/json")
        admin_oversized.endheaders()
        admin_oversized_response = admin_oversized.getresponse()
        assert admin_oversized_response.status == 413
        admin_oversized_response.read()
        admin_oversized.close()

        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps({"messages": [{"content": sentinel}]}).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {sentinel}",
                "Content-Type": "application/json",
            },
        )
        with pytest.raises(urllib.error.HTTPError) as upstream_error:
            urllib.request.urlopen(request, timeout=5)
        assert upstream_error.value.code == 502
        error_payload = json.loads(upstream_error.value.read().decode("utf-8"))
        assert error_payload == {
            "error": {
                "code": "upstream_transport_error",
                "route": "chat",
            }
        }
    finally:
        server.shutdown()
        server.server_close()

    captured = capsys.readouterr()
    assert sentinel not in captured.out
    assert sentinel not in captured.err


def test_model_bridge_admin_reload_keeps_targets_separate():
    module = load_model_bridge_module()
    token = "unit-token"
    initial = module.build_config(
        chat_target_base_url="https://chat.example.test/v1",
        chat_resolve_ip="1.1.1.1",
        embedding_target_base_url="https://embedding.example.test/v1",
        embedding_resolve_ip="2.2.2.2",
        timeout=30,
    )
    module.BridgeState.configure(initial, token)
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.ModelBridgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = read_json(f"{base_url}/health")
        assert health["status"] == "ok"
        assert "target_base_url" not in health
        assert health["chat_target_hash"] != health["embedding_target_hash"]

        try:
            read_json(f"{base_url}/admin/config")
            raise AssertionError("admin config should require a token")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        current = read_json(f"{base_url}/admin/config", token=token)
        assert current["chat_target_base_url"] == "https://chat.example.test/v1"
        assert current["embedding_target_base_url"] == "https://embedding.example.test/v1"

        old_snapshot = module.BridgeState.config()
        reloaded = post_json(
            f"{base_url}/admin/reload",
            {
                "chat_api_protocol": "anthropic",
                "chat_target_base_url": "https://chat.example.test/v2",
                "chat_resolve_ip": "3.3.3.3",
                "embedding_target_base_url": "https://embedding.example.test/v2",
                "embedding_resolve_ip": "4.4.4.4",
                "timeout": 45,
            },
            token=token,
        )

        assert reloaded["chat_target_base_url"] == "https://chat.example.test/v2"
        assert reloaded["chat_api_protocol"] == "anthropic"
        assert reloaded["embedding_target_base_url"] == "https://embedding.example.test/v2"
        assert reloaded["config_version"] != current["config_version"]
        assert old_snapshot.chat_target_base_url == "https://chat.example.test/v1"
        assert module.BridgeState.config().embedding_resolve_ip == "4.4.4.4"
    finally:
        server.shutdown()
        server.server_close()

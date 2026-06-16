from __future__ import annotations

import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path


def load_model_bridge_module():
    module_path = Path(__file__).resolve().parents[3] / "infra" / "model-bridge" / "model_bridge.py"
    spec = importlib.util.spec_from_file_location("model_bridge_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_model_bridge_routes_embeddings_to_embedding_target():
    module = load_model_bridge_module()

    assert module.route_for_path("/embeddings") == "embedding"
    assert module.route_for_path("/v1/embeddings?encoding=float") == "embedding"
    assert module.route_for_path("/chat/completions") == "chat"


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
                "chat_target_base_url": "https://chat.example.test/v2",
                "chat_resolve_ip": "3.3.3.3",
                "embedding_target_base_url": "https://embedding.example.test/v2",
                "embedding_resolve_ip": "4.4.4.4",
                "timeout": 45,
            },
            token=token,
        )

        assert reloaded["chat_target_base_url"] == "https://chat.example.test/v2"
        assert reloaded["embedding_target_base_url"] == "https://embedding.example.test/v2"
        assert reloaded["config_version"] != current["config_version"]
        assert old_snapshot.chat_target_base_url == "https://chat.example.test/v1"
        assert module.BridgeState.config().embedding_resolve_ip == "4.4.4.4"
    finally:
        server.shutdown()
        server.server_close()

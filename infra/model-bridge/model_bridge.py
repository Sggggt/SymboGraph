from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


_thread_local = threading.local()
_original_getaddrinfo = socket.getaddrinfo


def _thread_safe_getaddrinfo(*args, **kwargs):
    if not args:
        return _original_getaddrinfo(*args, **kwargs)
    host = args[0]
    overrides = getattr(_thread_local, "dns_overrides", None)
    if overrides and host in overrides:
        resolved_ip = overrides[host]
        if resolved_ip:
            port = args[1] if len(args) > 1 else 0
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolved_ip, port))]
    return _original_getaddrinfo(*args, **kwargs)


socket.getaddrinfo = _thread_safe_getaddrinfo


@dataclass(frozen=True)
class BridgeRouteConfig:
    chat_target_base_url: str
    chat_resolve_ip: str | None
    embedding_target_base_url: str
    embedding_resolve_ip: str | None
    timeout: int
    config_version: str
    updated_at: float


class BridgeState:
    _lock = threading.RLock()
    _config: BridgeRouteConfig | None = None
    _admin_token: str = ""
    _resolved_ip_cache: dict[tuple[str, str], tuple[str, float]] = {}

    @classmethod
    def configure(cls, config: BridgeRouteConfig, admin_token: str) -> None:
        with cls._lock:
            cls._config = config
            cls._admin_token = admin_token
            cls._resolved_ip_cache = {}

    @classmethod
    def config(cls) -> BridgeRouteConfig:
        with cls._lock:
            if cls._config is None:
                raise RuntimeError("Bridge is not configured")
            return cls._config

    @classmethod
    def admin_token(cls) -> str:
        with cls._lock:
            return cls._admin_token

    @classmethod
    def reload(cls, *, chat_target_base_url: str, chat_resolve_ip: str | None, embedding_target_base_url: str, embedding_resolve_ip: str | None, timeout: int | None = None) -> BridgeRouteConfig:
        with cls._lock:
            current = cls.config()
            next_config = build_config(
                chat_target_base_url=chat_target_base_url,
                chat_resolve_ip=chat_resolve_ip,
                embedding_target_base_url=embedding_target_base_url,
                embedding_resolve_ip=embedding_resolve_ip,
                timeout=int(timeout if timeout is not None else current.timeout),
            )
            cls._config = next_config
            cls._resolved_ip_cache = {}
            return next_config

    @classmethod
    def resolve_target_ip(cls, route: str, hostname: str, configured_resolve_ip: str | None) -> str | None:
        configured = (configured_resolve_ip or "").strip()
        if configured == "__none__":
            return None
        if configured:
            return configured

        cache_key = (route, hostname)
        now = time.time()
        with cls._lock:
            cached = cls._resolved_ip_cache.get(cache_key)
            if cached and cached[1] > now:
                return cached[0]

        ip = resolve_public_a_record(hostname)
        if ip:
            with cls._lock:
                cls._resolved_ip_cache[cache_key] = (ip, now + 300)
        return ip


def config_version_payload(*, chat_target_base_url: str, chat_resolve_ip: str | None, embedding_target_base_url: str, embedding_resolve_ip: str | None, timeout: int) -> dict[str, object]:
    return {
        "chat_target_base_url": chat_target_base_url.rstrip("/"),
        "chat_resolve_ip": chat_resolve_ip or "",
        "embedding_target_base_url": embedding_target_base_url.rstrip("/"),
        "embedding_resolve_ip": embedding_resolve_ip or "",
        "timeout": int(timeout),
    }


def stable_hash(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def target_hash(value: str) -> str:
    return hashlib.sha256((value or "").rstrip("/").encode("utf-8")).hexdigest()


def build_config(*, chat_target_base_url: str, chat_resolve_ip: str | None, embedding_target_base_url: str, embedding_resolve_ip: str | None, timeout: int) -> BridgeRouteConfig:
    chat_target = chat_target_base_url.rstrip("/")
    embedding_target = embedding_target_base_url.rstrip("/")
    if not chat_target:
        raise ValueError("chat_target_base_url is required")
    if not embedding_target:
        raise ValueError("embedding_target_base_url is required")
    version_payload = config_version_payload(
        chat_target_base_url=chat_target,
        chat_resolve_ip=chat_resolve_ip,
        embedding_target_base_url=embedding_target,
        embedding_resolve_ip=embedding_resolve_ip,
        timeout=timeout,
    )
    return BridgeRouteConfig(
        chat_target_base_url=chat_target,
        chat_resolve_ip=chat_resolve_ip or None,
        embedding_target_base_url=embedding_target,
        embedding_resolve_ip=embedding_resolve_ip or None,
        timeout=int(timeout),
        config_version=stable_hash(version_payload),
        updated_at=time.time(),
    )


def route_for_path(path: str) -> str:
    parsed = urlparse(path)
    normalized_path = parsed.path.rstrip("/").lower()
    if normalized_path.endswith("/embeddings"):
        return "embedding"
    return "chat"


def route_target(config: BridgeRouteConfig, route: str) -> tuple[str, str | None]:
    if route == "embedding":
        return config.embedding_target_base_url, config.embedding_resolve_ip
    return config.chat_target_base_url, config.chat_resolve_ip


def public_config(config: BridgeRouteConfig) -> dict[str, object]:
    return {
        "status": "ok",
        "config_version": config.config_version,
        "updated_at": config.updated_at,
        "chat_target_hash": target_hash(config.chat_target_base_url),
        "embedding_target_hash": target_hash(config.embedding_target_base_url),
        "chat_resolve_ip_configured": bool(config.chat_resolve_ip and config.chat_resolve_ip != "__none__"),
        "embedding_resolve_ip_configured": bool(config.embedding_resolve_ip and config.embedding_resolve_ip != "__none__"),
        "timeout": config.timeout,
        "routes": {
            "/chat/completions": "chat",
            "/embeddings": "embedding",
        },
    }


def admin_config(config: BridgeRouteConfig) -> dict[str, object]:
    return {
        **public_config(config),
        "chat_target_base_url": config.chat_target_base_url,
        "chat_resolve_ip": config.chat_resolve_ip or "",
        "embedding_target_base_url": config.embedding_target_base_url,
        "embedding_resolve_ip": config.embedding_resolve_ip or "",
    }


class ModelBridgeHandler(BaseHTTPRequestHandler):
    server_version = "CourseKGModelBridge/2.0"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, public_config(BridgeState.config()))
            return
        if self.path == "/admin/config":
            if not self._authorize_admin():
                return
            self._send_json(200, admin_config(BridgeState.config()))
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/admin/reload":
            self._handle_reload()
            return
        route = route_for_path(self.path)
        config = BridgeState.config()
        target_base_url, resolve_ip = route_target(config, route)
        target_url = target_base_url.rstrip("/") + self.path
        body_length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(body_length)
        try:
            try:
                status_code, response_body = self._forward_with_urllib(target_url, body, route=route, timeout=config.timeout, resolve_ip=resolve_ip)
            except Exception as urllib_exc:
                print(f"urllib forwarding failed for {route}, retrying with curl: {urllib_exc}", flush=True)
                status_code, response_body = self._forward_with_curl(target_url, body, route=route, timeout=config.timeout, resolve_ip=resolve_ip)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self._send_json(502, {"error": {"message": str(exc), "type": type(exc).__name__, "route": route}})
            return
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_body)

    def _handle_reload(self) -> None:
        if not self._authorize_admin():
            return
        body_length = int(self.headers.get("Content-Length") or "0")
        raw_body = self.rfile.read(body_length)
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            current = BridgeState.config()
            chat_resolve_ip = (
                _empty_to_none(payload.get("chat_resolve_ip"))
                if "chat_resolve_ip" in payload
                else current.chat_resolve_ip
            )
            embedding_resolve_ip = (
                _empty_to_none(payload.get("embedding_resolve_ip"))
                if "embedding_resolve_ip" in payload
                else current.embedding_resolve_ip
            )
            next_config = BridgeState.reload(
                chat_target_base_url=str(payload.get("chat_target_base_url") or current.chat_target_base_url),
                chat_resolve_ip=chat_resolve_ip,
                embedding_target_base_url=str(payload.get("embedding_target_base_url") or current.embedding_target_base_url),
                embedding_resolve_ip=embedding_resolve_ip,
                timeout=int(payload.get("timeout") or current.timeout),
            )
        except Exception as exc:
            self._send_json(400, {"error": {"message": str(exc), "type": type(exc).__name__}})
            return
        self._send_json(200, admin_config(next_config))

    def _authorize_admin(self) -> bool:
        token = BridgeState.admin_token()
        if not token:
            self._send_json(503, {"error": "admin_token_not_configured"})
            return False
        supplied = self.headers.get("X-Bridge-Admin-Token") or ""
        if not constant_time_equal(supplied, token):
            self._send_json(401, {"error": "unauthorized"})
            return False
        return True

    def _forward_with_urllib(self, target_url: str, body: bytes, *, route: str, timeout: int, resolve_ip: str | None) -> tuple[int, bytes]:
        parsed = urlparse(target_url)
        if not parsed.hostname:
            raise ValueError(f"Invalid target URL: {target_url}")

        auth_header = self.headers.get("Authorization")
        if not auth_header:
            raise ValueError("Missing Authorization header")

        resolved_ip = BridgeState.resolve_target_ip(route, parsed.hostname, resolve_ip)

        headers = {}
        for key, val in self.headers.items():
            if key.lower() in ("host", "content-length", "connection", "transfer-encoding"):
                continue
            headers[key] = val
        headers["Host"] = parsed.hostname

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        proxy_support = urllib.request.ProxyHandler({})
        https_handler = urllib.request.HTTPSHandler(context=ctx)
        opener = urllib.request.build_opener(proxy_support, https_handler)
        req = urllib.request.Request(target_url, data=body, headers=headers, method="POST")

        if resolved_ip:
            _thread_local.dns_overrides = {parsed.hostname: resolved_ip}
        try:
            with opener.open(req, timeout=timeout) as response:
                status_code = response.status
                response_body = response.read()
                return status_code, maybe_decompress(response_body)
        except urllib.error.HTTPError as e:
            return e.code, maybe_decompress(e.read())
        except Exception as e:
            raise RuntimeError(f"Urllib forwarding failed: {e}")
        finally:
            if hasattr(_thread_local, "dns_overrides"):
                delattr(_thread_local, "dns_overrides")

    def log_message(self, format: str, *args: object) -> None:
        if self.path != "/health":
            super().log_message(format, *args)

    def _forward_with_curl(self, target_url: str, body: bytes, *, route: str, timeout: int, resolve_ip: str | None) -> tuple[int, bytes]:
        parsed = urlparse(target_url)
        if not parsed.hostname:
            raise ValueError(f"Invalid target URL: {target_url}")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        auth_header = self.headers.get("Authorization")
        if not auth_header:
            raise ValueError("Missing Authorization header")

        cleanup_private_temp_dirs("coursekg-model-bridge-", max_age_seconds=3600)
        temp_dir = tempfile.mkdtemp(prefix="coursekg-model-bridge-")
        try:
            os.chmod(temp_dir, 0o700)
        except OSError:
            pass
        try:
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".json", dir=temp_dir) as body_file:
                body_file.write(body)
                body_path = body_file.name
            with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".json", dir=temp_dir) as output_file:
                output_path = output_file.name
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".curl", dir=temp_dir) as config_file:
                config_file.write(f'url = "{target_url}"\n')
                config_file.write("request = POST\n")
                config_file.write(f"connect-timeout = {min(timeout, 30)}\n")
                config_file.write(f"max-time = {timeout}\n")
                config_file.write("retry = 2\n")
                config_file.write("retry-delay = 1\n")
                config_file.write("retry-all-errors\n")
                config_file.write("insecure\n")
                resolved_ip = BridgeState.resolve_target_ip(route, parsed.hostname, resolve_ip)
                if resolved_ip:
                    config_file.write(f'resolve = "{parsed.hostname}:{port}:{resolved_ip}"\n')
                config_file.write(f'header = "Authorization: {auth_header}"\n')
                config_file.write('header = "Content-Type: application/json"\n')
                config_file.write(f'data-binary = "@{body_path.replace("\\", "/")}"\n')
                config_file.write(f'output = "{output_path.replace("\\", "/")}"\n')
                config_file.write('write-out = "%{http_code}"\n')
                config_path = config_file.name
            curl_binary = shutil.which("curl.exe") or shutil.which("curl")
            if not curl_binary:
                raise RuntimeError("curl is required by the model bridge but was not found")
            result = subprocess.run(
                [curl_binary, "-sS", "-K", config_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout + 10,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"curl exited with {result.returncode}")
            try:
                status_code = int(result.stdout.strip()[-3:])
            except ValueError as exc:
                raise RuntimeError(f"curl did not return a valid HTTP status: {result.stdout!r}") from exc
            with open(output_path, "rb") as output_file:
                response_body = output_file.read()
            return status_code, response_body or b"{}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _empty_to_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def constant_time_equal(left: str, right: str) -> bool:
    if len(left) != len(right):
        return False
    result = 0
    for left_char, right_char in zip(left.encode("utf-8"), right.encode("utf-8")):
        result |= left_char ^ right_char
    return result == 0


def maybe_decompress(response_body: bytes) -> bytes:
    if response_body.startswith(b"\x1f\x8b"):
        try:
            import gzip

            return gzip.decompress(response_body)
        except Exception as e:
            print(f"Failed to decompress response: {e}", flush=True)
    return response_body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target-base-url", default="")
    parser.add_argument("--resolve-ip", default="")
    parser.add_argument("--chat-target-base-url", default="")
    parser.add_argument("--embedding-target-base-url", default="")
    parser.add_argument("--chat-resolve-ip", default="")
    parser.add_argument("--embedding-resolve-ip", default="")
    parser.add_argument("--admin-token", default="")
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def resolve_public_a_record(hostname: str) -> str | None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    proxy_support = urllib.request.ProxyHandler({})
    https_handler = urllib.request.HTTPSHandler(context=ctx)
    opener = urllib.request.build_opener(proxy_support, https_handler)

    for resolver_url in (
        f"https://dns.alidns.com/resolve?name={hostname}&type=A",
        f"https://cloudflare-dns.com/dns-query?name={hostname}&type=A",
    ):
        try:
            request = urllib.request.Request(resolver_url, headers={"accept": "application/dns-json"})
            with opener.open(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for answer in payload.get("Answer", []):
                if int(answer.get("type", 0)) != 1:
                    continue
                value = str(answer.get("data", "")).strip()
                if is_public_ip(value):
                    return value
        except Exception:
            continue
    return None


def is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def cleanup_private_temp_dirs(prefix: str, *, max_age_seconds: int) -> None:
    temp_root = tempfile.gettempdir()
    cutoff = time.time() - max_age_seconds
    try:
        names = os.listdir(temp_root)
    except OSError:
        return
    for name in names:
        if not name.startswith(prefix):
            continue
        path = os.path.join(temp_root, name)
        try:
            if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def main() -> None:
    args = parse_args()
    cleanup_private_temp_dirs("coursekg-model-bridge-", max_age_seconds=0)
    fallback_target = args.target_base_url.rstrip("/")
    chat_target = (args.chat_target_base_url or fallback_target).rstrip("/")
    embedding_target = (args.embedding_target_base_url or fallback_target).rstrip("/")
    chat_resolve_ip = args.chat_resolve_ip or args.resolve_ip or None
    embedding_resolve_ip = args.embedding_resolve_ip or args.resolve_ip or None
    admin_token = args.admin_token or os.getenv("MODEL_BRIDGE_ADMIN_TOKEN", "")
    config = build_config(
        chat_target_base_url=chat_target,
        chat_resolve_ip=chat_resolve_ip,
        embedding_target_base_url=embedding_target,
        embedding_resolve_ip=embedding_resolve_ip,
        timeout=args.timeout,
    )
    BridgeState.configure(config, admin_token)
    server = ThreadingHTTPServer((args.host, args.port), ModelBridgeHandler)
    print(
        "Model bridge listening on "
        f"http://{args.host}:{args.port} "
        f"chat={target_hash(config.chat_target_base_url)[:12]} "
        f"embedding={target_hash(config.embedding_target_base_url)[:12]} "
        f"version={config.config_version[:12]}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

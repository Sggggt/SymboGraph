from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import ipaddress
import json
import os
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode, urlparse


_thread_local = threading.local()
_original_getaddrinfo = socket.getaddrinfo
MODEL_PROXY_ROUTES = {
    "/chat/completions": "chat",
    "/v1/messages": "chat_anthropic",
    "/embeddings": "embedding",
}
MODEL_API_PROTOCOLS = frozenset({"openai", "anthropic"})
EMBEDDING_API_PROTOCOLS = frozenset({"openai"})
ANTHROPIC_VERSION = "2023-06-01"
MAX_MODEL_REQUEST_BODY_BYTES = 16 * 1024 * 1024
MAX_MODEL_RESPONSE_BODY_BYTES = 32 * 1024 * 1024
MAX_ADMIN_RELOAD_BODY_BYTES = 64 * 1024
MAX_DOH_RESPONSE_BODY_BYTES = 256 * 1024
MAX_DOH_ANSWER_RECORDS = 256
DOH_RESOLVERS = (
    (
        "dns.alidns.com",
        "/resolve",
        ("223.5.5.5", "223.6.6.6"),
    ),
    (
        "cloudflare-dns.com",
        "/dns-query",
        ("1.1.1.1", "1.0.0.1"),
    ),
)
KNOWN_UNSAFE_ADMIN_TOKENS = frozenset(
    {
        "change-me",
        "changeme",
        "default",
        "local-model-bridge-admin",
        "model-bridge-admin-token",
    }
)


class RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail closed instead of forwarding credentials across redirects."""

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        del msg, newurl
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            "upstream_redirect_rejected",
            headers,
            fp,
        )


def _thread_safe_getaddrinfo(*args, **kwargs):
    if not args:
        return _original_getaddrinfo(*args, **kwargs)
    host = args[0]
    overrides = getattr(_thread_local, "dns_overrides", None)
    if overrides and host in overrides:
        resolved_ip = overrides[host]
        if not resolved_ip or not is_public_ip(resolved_ip):
            raise socket.gaierror("provider DNS override is not globally routable")
        ip_value = ipaddress.ip_address(resolved_ip)
        port = args[1] if len(args) > 1 else 0
        if ip_value.version == 6:
            return [
                (
                    socket.AF_INET6,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    (str(ip_value), port, 0, 0),
                )
            ]
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (str(ip_value), port),
            )
        ]
    if getattr(_thread_local, "forbid_system_dns", False):
        raise socket.gaierror("system DNS is disabled for this HTTPS hop")
    return _original_getaddrinfo(*args, **kwargs)


socket.getaddrinfo = _thread_safe_getaddrinfo


@dataclass(frozen=True)
class BridgeRouteConfig:
    chat_api_protocol: str
    embedding_api_protocol: str
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
        verified_token = verified_admin_token(admin_token)
        with cls._lock:
            cls._config = config
            cls._admin_token = verified_token
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
    def reload(cls, *, chat_api_protocol: str, embedding_api_protocol: str = "openai", chat_target_base_url: str, chat_resolve_ip: str | None, embedding_target_base_url: str, embedding_resolve_ip: str | None, timeout: int | None = None) -> BridgeRouteConfig:
        with cls._lock:
            current = cls.config()
            next_config = build_config(
                chat_api_protocol=chat_api_protocol,
                embedding_api_protocol=embedding_api_protocol,
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
    def resolve_target_ip(cls, route: str, hostname: str, configured_resolve_ip: str | None) -> str:
        configured = (configured_resolve_ip or "").strip()
        if configured:
            if not is_public_ip(configured):
                raise RuntimeError("Configured provider IP is not globally routable")
            return configured

        try:
            literal_ip = ipaddress.ip_address(hostname)
        except ValueError:
            literal_ip = None
        if literal_ip is not None:
            if not is_public_ip(str(literal_ip)):
                raise RuntimeError("Provider hostname is not globally routable")
            return str(literal_ip)

        cache_key = (route, hostname)
        now = time.time()
        with cls._lock:
            cached = cls._resolved_ip_cache.get(cache_key)
            if cached and cached[1] > now:
                return cached[0]

        ip = resolve_public_a_record(hostname)
        if not ip:
            raise RuntimeError("Provider hostname has no verified public A record")
        with cls._lock:
            cls._resolved_ip_cache[cache_key] = (ip, now + 300)
        return ip


def config_version_payload(*, chat_api_protocol: str, embedding_api_protocol: str = "openai", chat_target_base_url: str, chat_resolve_ip: str | None, embedding_target_base_url: str, embedding_resolve_ip: str | None, timeout: int) -> dict[str, object]:
    return {
        "chat_api_protocol": chat_api_protocol,
        "embedding_api_protocol": embedding_api_protocol,
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


def verified_tls_context() -> ssl.SSLContext:
    """Return a fail-closed client context for every bridge HTTPS hop."""

    context = ssl.create_default_context()
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        raise RuntimeError(
            "Model bridge TLS verification is unavailable: CA and hostname "
            "verification are both required"
        )
    return context


def build_verified_https_opener(
    context: ssl.SSLContext,
) -> urllib.request.OpenerDirector:
    """Build an HTTPS opener with proxies and every redirect disabled."""

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        RejectRedirectHandler(),
        urllib.request.HTTPSHandler(context=context),
    )
    # OpenerDirector otherwise injects Python-urllib/<version> as User-Agent
    # during HTTPSHandler.do_request_.  Provider requests use a closed header
    # contract: only the caller-validated application headers plus Host and
    # Content-Length derived by the HTTP stack are permitted.
    opener.addheaders = []
    return opener


def verified_admin_token(value: str) -> str:
    token = str(value or "")
    validation_value = token
    if (
        len(validation_value) >= 2
        and validation_value[0] in {'"', "'"}
        and validation_value[-1] == validation_value[0]
    ):
        validation_value = validation_value[1:-1]
    if (
        not token
        or token != token.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in token)
        or not validation_value
        or validation_value != validation_value.strip()
        or any(
            ord(char) < 32 or ord(char) == 127
            for char in validation_value
        )
        or validation_value.casefold() in KNOWN_UNSAFE_ADMIN_TOKENS
    ):
        raise ValueError(
            "MODEL_BRIDGE_ADMIN_TOKEN must be explicit and non-default"
        )
    return token


def running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def verified_bind_host(
    value: str,
    *,
    allow_private_container_bind: bool = False,
) -> str:
    host = str(value or "").strip()
    try:
        ip_value = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("model bridge bind host must be a literal loopback IP") from exc
    private_container_any = bool(
        allow_private_container_bind
        and running_in_container()
        and isinstance(ip_value, ipaddress.IPv4Address)
        and ip_value == ipaddress.IPv4Address("0.0.0.0")
    )
    if not ip_value.is_loopback and not private_container_any:
        raise ValueError("model bridge bind host must be a literal loopback IP")
    return str(ip_value)


def verified_provider_base_url(value: str, *, field: str) -> str:
    target = (value or "").rstrip("/")
    if (
        not target
        or target != target.strip()
        or "\\" in target
        or any(char.isspace() for char in target)
        or any(ord(char) < 32 or ord(char) == 127 for char in target)
        or "?" in target
        or "#" in target
    ):
        raise ValueError(f"{field} is invalid")
    parsed = urlparse(target)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.params
    ):
        raise ValueError(f"{field} must be one credential-free HTTPS base URL")
    hostname = parsed.hostname.rstrip(".")
    if not hostname or not hostname.isascii() or "%" in hostname:
        raise ValueError(f"{field} hostname must use canonical ASCII syntax")
    normalized_hostname = hostname.casefold()
    if normalized_hostname == "localhost" or normalized_hostname.endswith(
        ".localhost"
    ):
        raise ValueError(f"{field} cannot target localhost")
    try:
        literal_ip = ipaddress.ip_address(normalized_hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if not is_public_ip(str(literal_ip)):
            raise ValueError(f"{field} cannot target a non-public IP address")
    else:
        labels = normalized_hostname.split(".")

        def numeric_address_token(label: str) -> bool:
            if label.isdigit():
                return True
            return bool(
                label.startswith("0x")
                and len(label) > 2
                and all(char in "0123456789abcdef" for char in label[2:])
            )

        if normalized_hostname.isdigit() or all(
            numeric_address_token(label) for label in labels
        ):
            raise ValueError(f"{field} uses a non-canonical IP spelling")
        if len(normalized_hostname) > 253 or any(
            not label or len(label) > 63 for label in labels
        ):
            raise ValueError(f"{field} hostname is not canonical DNS syntax")
        for label in labels:
            if (
                not label[0].isalnum()
                or not label[-1].isalnum()
                or any(not (char.isalnum() or char == "-") for char in label)
            ):
                raise ValueError(f"{field} hostname is not canonical DNS syntax")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid port") from exc
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"{field} port is outside 1..65535")
    return target


def verified_resolve_ip(value: str | None, *, field: str) -> str | None:
    configured = (value or "").strip()
    if not configured or configured == "__none__":
        return None
    if not is_public_ip(configured):
        raise ValueError(f"{field} must be one public IP address")
    return configured


def verified_provider_endpoint_base_url(
    value: str,
    *,
    protocol: str,
    purpose: str,
    field: str,
) -> str:
    target = verified_provider_base_url(value, field=field)
    path = (urlparse(target).path or "").rstrip("/").casefold()
    fixed_route = (
        "/embeddings"
        if purpose == "embedding"
        else ("/v1/messages" if protocol == "anthropic" else "/chat/completions")
    )
    if path.endswith(fixed_route):
        raise ValueError(f"{field} must not include the fixed request route")
    if protocol == "anthropic" and path.endswith("/v1"):
        raise ValueError(f"{field} must not end in /v1 for Anthropic Messages")
    return target


def build_config(*, chat_api_protocol: str = "openai", embedding_api_protocol: str = "openai", chat_target_base_url: str, chat_resolve_ip: str | None, embedding_target_base_url: str, embedding_resolve_ip: str | None, timeout: int) -> BridgeRouteConfig:
    if chat_api_protocol not in MODEL_API_PROTOCOLS:
        raise ValueError("chat_api_protocol must be openai or anthropic")
    if embedding_api_protocol not in EMBEDDING_API_PROTOCOLS:
        raise ValueError("embedding_api_protocol must be openai")
    chat_target = verified_provider_endpoint_base_url(
        chat_target_base_url,
        protocol=chat_api_protocol,
        purpose="chat",
        field="chat_target_base_url",
    )
    embedding_target = verified_provider_endpoint_base_url(
        embedding_target_base_url,
        protocol=embedding_api_protocol,
        purpose="embedding",
        field="embedding_target_base_url",
    )
    verified_chat_resolve_ip = verified_resolve_ip(
        chat_resolve_ip,
        field="chat_resolve_ip",
    )
    verified_embedding_resolve_ip = verified_resolve_ip(
        embedding_resolve_ip,
        field="embedding_resolve_ip",
    )
    version_payload = config_version_payload(
        chat_api_protocol=chat_api_protocol,
        embedding_api_protocol=embedding_api_protocol,
        chat_target_base_url=chat_target,
        chat_resolve_ip=verified_chat_resolve_ip,
        embedding_target_base_url=embedding_target,
        embedding_resolve_ip=verified_embedding_resolve_ip,
        timeout=timeout,
    )
    return BridgeRouteConfig(
        chat_api_protocol=chat_api_protocol,
        embedding_api_protocol=embedding_api_protocol,
        chat_target_base_url=chat_target,
        chat_resolve_ip=verified_chat_resolve_ip,
        embedding_target_base_url=embedding_target,
        embedding_resolve_ip=verified_embedding_resolve_ip,
        timeout=int(timeout),
        config_version=stable_hash(version_payload),
        updated_at=time.time(),
    )


def route_for_path(path: str) -> str | None:
    # ``BaseHTTPRequestHandler.path`` can contain an absolute-form request
    # target when a client treats the bridge as an HTTP proxy.  Never reduce
    # such a target (or a network-path reference) to its parsed path: only the
    # two exact origin-form targets are part of the bridge protocol.
    if path not in MODEL_PROXY_ROUTES:
        return None
    parsed = urlparse(path)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.params
        or parsed.path != path
        or not path.startswith("/")
        or path.startswith("//")
    ):
        return None
    return MODEL_PROXY_ROUTES[path]


def route_target(config: BridgeRouteConfig, route: str) -> tuple[str, str | None]:
    if route == "embedding":
        return config.embedding_target_base_url, config.embedding_resolve_ip
    return config.chat_target_base_url, config.chat_resolve_ip


def route_matches_protocol(config: BridgeRouteConfig, route: str) -> bool:
    if route == "embedding":
        return config.embedding_api_protocol == "openai"
    expected_route = "chat_anthropic" if config.chat_api_protocol == "anthropic" else "chat"
    return route == expected_route


def validated_client_auth_headers(headers, *, route: str) -> dict[str, str]:
    if route == "chat_anthropic":
        authorization = str(headers.get("Authorization") or "")
        version = str(headers.get("Anthropic-Version") or "")
        if headers.get("X-Api-Key"):
            raise ValueError("legacy Anthropic API key header is not allowed")
        if (
            not authorization.startswith("Bearer ")
            or not authorization.removeprefix("Bearer ")
            or authorization != authorization.strip()
            or len(authorization.encode("utf-8")) > 16 * 1024
            or any(
                ord(char) < 32 or ord(char) == 127
                for char in authorization
            )
            or version != ANTHROPIC_VERSION
        ):
            raise ValueError("invalid Anthropic authentication headers")
        return {
            "Authorization": authorization,
            "Anthropic-Version": ANTHROPIC_VERSION,
        }
    authorization = str(headers.get("Authorization") or "")
    if headers.get("X-Api-Key") or headers.get("Anthropic-Version"):
        raise ValueError("cross-protocol Anthropic header")
    if (
        not authorization
        or authorization != authorization.strip()
        or len(authorization.encode("utf-8")) > 16 * 1024
        or any(ord(char) < 32 or ord(char) == 127 for char in authorization)
    ):
        raise ValueError("invalid Authorization header")
    return {"Authorization": authorization}


def public_config(config: BridgeRouteConfig) -> dict[str, object]:
    chat_route = (
        "/v1/messages"
        if config.chat_api_protocol == "anthropic"
        else "/chat/completions"
    )
    return {
        "status": "ok",
        "config_version": config.config_version,
        "updated_at": config.updated_at,
        "chat_target_hash": target_hash(config.chat_target_base_url),
        "embedding_target_hash": target_hash(config.embedding_target_base_url),
        "chat_resolve_ip_configured": bool(config.chat_resolve_ip and config.chat_resolve_ip != "__none__"),
        "embedding_resolve_ip_configured": bool(config.embedding_resolve_ip and config.embedding_resolve_ip != "__none__"),
        "timeout": config.timeout,
        "chat_api_protocol": config.chat_api_protocol,
        "embedding_api_protocol": config.embedding_api_protocol,
        "routes": {
            chat_route: (
                "chat_anthropic"
                if config.chat_api_protocol == "anthropic"
                else "chat"
            ),
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
        if route is None:
            self._send_json(404, {"error": "not_found"})
            return
        config = BridgeState.config()
        if not route_matches_protocol(config, route):
            self._send_json(404, {"error": "not_found"})
            return
        try:
            validated_client_auth_headers(self.headers, route=route)
        except ValueError:
            self._send_json(401, {"error": "invalid_provider_auth"})
            return
        try:
            body_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": "invalid_content_length"})
            return
        if body_length < 0 or body_length > MAX_MODEL_REQUEST_BODY_BYTES:
            self._send_json(413, {"error": "request_body_too_large"})
            return
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(415, {"error": "application_json_required"})
            return
        target_base_url, resolve_ip = route_target(config, route)
        target_url = target_base_url.rstrip("/") + self.path
        body = self.rfile.read(body_length)
        try:
            request_payload = _strict_json_object(
                body,
                error_message="Model bridge request was not strict JSON",
            )
        except RuntimeError:
            self._send_json(400, {"error": "invalid_json_body"})
            return
        try:
            status_code, response_body = self._forward_with_urllib(
                target_url,
                body,
                route=route,
                timeout=config.timeout,
                resolve_ip=resolve_ip,
            )
        except Exception as exc:
            print(
                f"model_bridge_upstream_failure route={route} "
                f"error_type={type(exc).__name__}",
                flush=True,
            )
            self._send_json(
                502,
                {
                    "error": {
                        "code": "upstream_transport_error",
                        "route": route,
                    }
                },
            )
            return
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_body)

    def _handle_reload(self) -> None:
        if not self._authorize_admin():
            return
        try:
            body_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._send_json(400, {"error": {"code": "invalid_content_length"}})
            return
        if body_length < 0 or body_length > MAX_ADMIN_RELOAD_BODY_BYTES:
            self._send_json(413, {"error": {"code": "admin_body_too_large"}})
            return
        content_type = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            self._send_json(415, {"error": {"code": "application_json_required"}})
            return
        raw_body = self.rfile.read(body_length)
        try:
            payload = _strict_json_object(
                raw_body or b"{}",
                error_message="Admin reload payload was not strict JSON",
            )
            current = BridgeState.config()
            chat_api_protocol = str(
                payload.get("chat_api_protocol")
                if "chat_api_protocol" in payload
                else current.chat_api_protocol
            )
            embedding_api_protocol = str(
                payload.get("embedding_api_protocol")
                if "embedding_api_protocol" in payload
                else current.embedding_api_protocol
            )
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
                chat_api_protocol=chat_api_protocol,
                embedding_api_protocol=embedding_api_protocol,
                chat_target_base_url=str(payload.get("chat_target_base_url") or current.chat_target_base_url),
                chat_resolve_ip=chat_resolve_ip,
                embedding_target_base_url=str(payload.get("embedding_target_base_url") or current.embedding_target_base_url),
                embedding_resolve_ip=embedding_resolve_ip,
                timeout=int(payload.get("timeout") or current.timeout),
            )
        except Exception:
            self._send_json(400, {"error": {"code": "invalid_admin_reload_payload"}})
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

        credential_headers = validated_client_auth_headers(
            self.headers,
            route=route,
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Host": parsed.netloc,
        }
        headers.update(credential_headers)
        resolved_ip = BridgeState.resolve_target_ip(route, parsed.hostname, resolve_ip)

        ctx = verified_tls_context()

        opener = build_verified_https_opener(ctx)
        req = urllib.request.Request(target_url, data=body, headers=headers, method="POST")

        _thread_local.dns_overrides = {parsed.hostname: resolved_ip}
        try:
            with opener.open(req, timeout=timeout) as response:
                status_code = response.status
                response_headers = response.headers
                content_type = str(response_headers.get("Content-Type", ""))
                if not _is_provider_json_content_type(content_type):
                    raise RuntimeError("Upstream response content type was not JSON")
                content_encoding = str(response_headers.get("Content-Encoding", ""))
                if content_encoding.strip().casefold() not in {"", "identity"}:
                    raise RuntimeError("Upstream response content encoding was unsupported")
                declared_length = str(response_headers.get("Content-Length", "")).strip()
                if declared_length:
                    if not declared_length.isascii() or not declared_length.isdecimal():
                        raise RuntimeError("Upstream response content length was invalid")
                    if int(declared_length) > MAX_MODEL_RESPONSE_BODY_BYTES:
                        raise RuntimeError("Upstream response exceeded the hard byte bound")
                response_body = response.read(MAX_MODEL_RESPONSE_BODY_BYTES + 1)
                if declared_length and len(response_body) != int(declared_length):
                    raise RuntimeError("Upstream response content length did not match the body")
                return status_code, validated_json_response_body(response_body)
        except urllib.error.HTTPError as e:
            upstream_status = int(e.code)
            e.close()
            if 300 <= upstream_status < 400:
                return (
                    502,
                    json.dumps(
                        {
                            "error": {
                                "code": "upstream_redirect_rejected",
                                "status": upstream_status,
                            }
                        },
                        separators=(",", ":"),
                    ).encode("utf-8"),
                )
            return (
                upstream_status,
                json.dumps(
                    {
                        "error": {
                            "code": "upstream_http_error",
                            "status": upstream_status,
                        }
                    },
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        except Exception:
            raise RuntimeError("Verified upstream HTTPS forwarding failed") from None
        finally:
            if hasattr(_thread_local, "dns_overrides"):
                delattr(_thread_local, "dns_overrides")

    def log_message(self, format: str, *args: object) -> None:
        # Do not emit request targets, headers, bodies, provider responses, or
        # credential-bearing exception text through BaseHTTPRequestHandler.
        return

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
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def maybe_decompress(response_body: bytes) -> bytes:
    if response_body.startswith(b"\x1f\x8b"):
        try:
            import gzip

            with gzip.GzipFile(fileobj=io.BytesIO(response_body)) as stream:
                decoded = stream.read(MAX_MODEL_RESPONSE_BODY_BYTES + 1)
            if len(decoded) > MAX_MODEL_RESPONSE_BODY_BYTES:
                raise RuntimeError("Upstream response exceeded the hard byte bound")
            return decoded
        except Exception:
            raise RuntimeError("Upstream response decompression failed") from None
    return response_body


def validated_json_response_body(response_body: bytes) -> bytes:
    if len(response_body) > MAX_MODEL_RESPONSE_BODY_BYTES:
        raise RuntimeError("Upstream response exceeded the hard byte bound")
    decoded = maybe_decompress(response_body)
    if len(decoded) > MAX_MODEL_RESPONSE_BODY_BYTES:
        raise RuntimeError("Upstream response exceeded the hard byte bound")
    _strict_json_object(decoded, error_message="Upstream response was not strict JSON")
    return decoded


def _is_provider_json_content_type(value: str) -> bool:
    media_type = str(value or "").split(";", 1)[0].strip().casefold()
    return bool(
        media_type == "application/json"
        or (media_type.startswith("application/") and media_type.endswith("+json"))
    )


def _strict_json_object(raw: bytes, *, error_message: str) -> dict:
    def reject_constant(_value: str):
        raise ValueError("non-finite JSON constant")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise RuntimeError(error_message) from None
    if not isinstance(payload, dict):
        raise RuntimeError(error_message)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--allow-private-container-bind",
        action="store_true",
        help=(
            "Allow 0.0.0.0 only from inside a container. The service must remain "
            "on a private Docker network with no published port."
        ),
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target-base-url", default="")
    parser.add_argument("--resolve-ip", default="")
    parser.add_argument("--chat-target-base-url", default="")
    parser.add_argument(
        "--chat-api-protocol",
        choices=sorted(MODEL_API_PROTOCOLS),
        default="openai",
    )
    parser.add_argument("--embedding-target-base-url", default="")
    parser.add_argument(
        "--embedding-api-protocol",
        choices=sorted(EMBEDDING_API_PROTOCOLS),
        default="openai",
    )
    parser.add_argument("--chat-resolve-ip", default="")
    parser.add_argument("--embedding-resolve-ip", default="")
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def resolve_public_a_record(hostname: str) -> str | None:
    ctx = verified_tls_context()
    opener = build_verified_https_opener(ctx)

    query = urlencode({"name": hostname, "type": "A"})
    for resolver_hostname, resolver_path, resolver_ips in DOH_RESOLVERS:
        resolver_url = f"https://{resolver_hostname}{resolver_path}?{query}"
        for resolver_ip in resolver_ips:
            if not is_public_ip(resolver_ip):
                continue
            request = urllib.request.Request(
                resolver_url,
                headers={
                    "Accept": "application/dns-json",
                    "Accept-Encoding": "identity",
                },
            )
            previous_overrides = getattr(_thread_local, "dns_overrides", None)
            previous_forbid_system_dns = getattr(
                _thread_local,
                "forbid_system_dns",
                False,
            )
            _thread_local.dns_overrides = {resolver_hostname: resolver_ip}
            _thread_local.forbid_system_dns = True
            try:
                with opener.open(request, timeout=5) as response:
                    status_code = getattr(response, "status", None)
                    if type(status_code) is not int:
                        status_code = response.getcode()
                    if status_code != 200:
                        continue
                    headers = response.headers
                    content_type = str(headers.get("Content-Type", ""))
                    content_encoding = str(headers.get("Content-Encoding", ""))
                    if not _is_doh_json_content_type(content_type):
                        continue
                    if content_encoding.strip().casefold() not in {"", "identity"}:
                        continue
                    response_body = response.read(
                        MAX_DOH_RESPONSE_BODY_BYTES + 1
                    )
                answer = _validated_doh_public_a_response(response_body)
                if answer is not None:
                    return answer
            except Exception:
                continue
            finally:
                if previous_overrides is None:
                    if hasattr(_thread_local, "dns_overrides"):
                        delattr(_thread_local, "dns_overrides")
                else:
                    _thread_local.dns_overrides = previous_overrides
                _thread_local.forbid_system_dns = previous_forbid_system_dns
    return None


def _is_doh_json_content_type(value: str) -> bool:
    parts = [part.strip().casefold() for part in value.split(";")]
    return bool(
        parts
        and parts[0] in {"application/dns-json", "application/json"}
        and (
            len(parts) == 1
            or (len(parts) == 2 and parts[1] == "charset=utf-8")
        )
    )


def _validated_doh_public_a_response(response_body: bytes) -> str | None:
    if len(response_body) > MAX_DOH_RESPONSE_BODY_BYTES:
        raise ValueError("DoH response exceeded the hard byte bound")

    def reject_constant(_value: str):
        raise ValueError("DoH response used a non-finite JSON constant")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DoH response used a duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            response_body.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("DoH response was not strict JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("DoH response was not one JSON object")
    if type(payload.get("Status")) is not int or payload["Status"] != 0:
        raise ValueError("DoH response status was not successful")
    if payload.get("TC") is not False:
        raise ValueError("DoH response was truncated or omitted TC")
    answers = payload.get("Answer")
    if not isinstance(answers, list) or len(answers) > MAX_DOH_ANSWER_RECORDS:
        raise ValueError("DoH response answer count was invalid")
    if not answers:
        return None

    public_answers: list[str] = []
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("DoH answer was not an object")
        answer_type = answer.get("type")
        if (
            type(answer_type) is not int
            or not 1 <= answer_type <= 65535
        ):
            raise ValueError("DoH answer type was invalid")
        if answer_type != 1:
            continue
        value = answer.get("data")
        if not isinstance(value, str) or value != value.strip():
            raise ValueError("DoH A record data was invalid")
        try:
            ip_value = ipaddress.ip_address(value)
        except ValueError:
            raise ValueError("DoH A record data was not an IP address") from None
        if (
            not isinstance(ip_value, ipaddress.IPv4Address)
            or str(ip_value) != value
            or not is_public_ip(value)
        ):
            raise ValueError("DoH A record was not canonical public unicast IPv4")
        public_answers.append(value)
    return public_answers[0] if public_answers else None


def is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return False
    return bool(
        ip.is_global
        and not ip.is_private
        and not ip.is_loopback
        and not ip.is_link_local
        and not ip.is_multicast
        and not ip.is_reserved
        and not ip.is_unspecified
        and not getattr(ip, "is_site_local", False)
    )


def main() -> None:
    args = parse_args()
    bind_host = verified_bind_host(
        args.host,
        allow_private_container_bind=bool(
            getattr(args, "allow_private_container_bind", False)
        ),
    )
    fallback_target = args.target_base_url.rstrip("/")
    chat_target = (args.chat_target_base_url or fallback_target).rstrip("/")
    embedding_target = (args.embedding_target_base_url or fallback_target).rstrip("/")
    chat_resolve_ip = args.chat_resolve_ip or args.resolve_ip or None
    embedding_resolve_ip = args.embedding_resolve_ip or args.resolve_ip or None
    admin_token = os.getenv("MODEL_BRIDGE_ADMIN_TOKEN", "")
    config = build_config(
        chat_api_protocol=getattr(args, "chat_api_protocol", "openai"),
        embedding_api_protocol=getattr(args, "embedding_api_protocol", "openai"),
        chat_target_base_url=chat_target,
        chat_resolve_ip=chat_resolve_ip,
        embedding_target_base_url=embedding_target,
        embedding_resolve_ip=embedding_resolve_ip,
        timeout=args.timeout,
    )
    BridgeState.configure(config, admin_token)
    server = ThreadingHTTPServer((bind_host, args.port), ModelBridgeHandler)
    print(
        "Model bridge listening on "
        f"http://{bind_host}:{args.port} "
        f"chat_protocol={config.chat_api_protocol} "
        f"embedding_protocol={config.embedding_api_protocol} "
        f"chat={target_hash(config.chat_target_base_url)[:12]} "
        f"embedding={target_hash(config.embedding_target_base_url)[:12]} "
        f"version={config.config_version[:12]}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

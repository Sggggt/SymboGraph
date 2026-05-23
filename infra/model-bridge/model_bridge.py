from __future__ import annotations

import argparse
import ipaddress
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


class BridgeConfig:
    target_base_url: str = ""
    resolve_ip: str | None = None
    timeout: int = 180
    resolved_ip_cache: tuple[str, float] | None = None


class ModelBridgeHandler(BaseHTTPRequestHandler):
    server_version = "CourseKGModelBridge/1.0"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "target_base_url": BridgeConfig.target_base_url})
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        target_url = BridgeConfig.target_base_url.rstrip("/") + self.path
        body_length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(body_length)
        try:
            status_code, response_body = self._forward_with_curl(target_url, body)
        except Exception as exc:
            self._send_json(502, {"error": {"message": str(exc), "type": type(exc).__name__}})
            return
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        if self.path != "/health":
            super().log_message(format, *args)

    def _forward_with_curl(self, target_url: str, body: bytes) -> tuple[int, bytes]:
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
                config_file.write(f"connect-timeout = {min(BridgeConfig.timeout, 30)}\n")
                config_file.write(f"max-time = {BridgeConfig.timeout}\n")
                config_file.write("retry = 2\n")
                config_file.write("retry-delay = 1\n")
                config_file.write("retry-all-errors\n")
                resolve_ip = resolve_target_ip(parsed.hostname)
                if resolve_ip:
                    config_file.write(f'resolve = "{parsed.hostname}:{port}:{resolve_ip}"\n')
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
                timeout=BridgeConfig.timeout + 10,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or f"curl exited with {result.returncode}")
            try:
                status_code = int(result.stdout.strip()[-3:])
            except ValueError as exc:
                raise RuntimeError(f"curl did not return a valid HTTP status: {result.stdout!r}") from exc
            with open(output_path, "rb") as output_file:
                response_body = output_file.read()
            if not response_body:
                response_body = b"{}"
            return status_code, response_body
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target-base-url", required=True)
    parser.add_argument("--resolve-ip", default="")
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def resolve_target_ip(hostname: str) -> str | None:
    configured = (BridgeConfig.resolve_ip or "").strip()
    if configured and configured != "__none__":
        return configured

    cached = BridgeConfig.resolved_ip_cache
    now = time.time()
    if cached and cached[1] > now:
        return cached[0]

    ip = resolve_public_a_record(hostname)
    BridgeConfig.resolved_ip_cache = (ip, now + 300)
    return ip


def resolve_public_a_record(hostname: str) -> str | None:
    for resolver_url in (
        f"https://dns.alidns.com/resolve?name={hostname}&type=A",
        f"https://cloudflare-dns.com/dns-query?name={hostname}&type=A",
    ):
        try:
            request = urllib.request.Request(resolver_url, headers={"accept": "application/dns-json"})
            with urllib.request.urlopen(request, timeout=5) as response:
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
    BridgeConfig.target_base_url = args.target_base_url.rstrip("/")
    BridgeConfig.resolve_ip = args.resolve_ip or None
    BridgeConfig.timeout = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), ModelBridgeHandler)
    print(f"Model bridge listening on http://{args.host}:{args.port} -> {BridgeConfig.target_base_url}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

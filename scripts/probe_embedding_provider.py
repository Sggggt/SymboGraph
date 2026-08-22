from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from _context_graph_maintenance import write_report


PROBE_PROTOCOL_VERSION = "embedding_provider_probe_v1"
DEFAULT_PROBE_TEXT = "SymboGraph 向量模型连通性测试"
MAX_PROBE_TEXT_CHARS = 512
MAX_PROBE_TEXT_UTF8_BYTES = 2048
MAX_BRIDGE_PROBE_RESPONSE_BYTES = 1024 * 1024
SAFE_BRIDGE_RESPONSE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
SAFE_BRIDGE_SYNC_FIELDS = frozenset(
    {
        "attempted",
        "ok",
        "config_version",
        "chat_target_hash",
        "embedding_target_hash",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or execute one content-free audited embedding provider "
            "connectivity probe. Execute only inside course-kg-api."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Issue one real embedding request. Omit for zero-network dry-run.",
    )
    parser.add_argument(
        "--arm",
        choices=("provider", "bridge"),
        default="provider",
        help=(
            "provider uses production EmbeddingProvider with bounded retries; "
            "bridge sends one direct request to the local Docker bridge."
        ),
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_PROBE_TEXT,
        help="Probe text sent only on --execute; reports retain only length and SHA-256.",
    )
    parser.add_argument(
        "--text-type",
        choices=("query", "document"),
        default="query",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="Overall probe timeout in seconds (1..600).",
    )
    return parser.parse_args()


def probe_input_card(text: str, *, text_type: str) -> dict[str, Any]:
    normalized = str(text or "")
    if not normalized.strip():
        raise ValueError("Probe text must not be empty")
    if len(normalized) > MAX_PROBE_TEXT_CHARS:
        raise ValueError("Probe text exceeds the character bound")
    encoded = normalized.encode("utf-8")
    if len(encoded) > MAX_PROBE_TEXT_UTF8_BYTES:
        raise ValueError("Probe text exceeds the UTF-8 byte bound")
    if any(ord(character) < 32 and character not in "\t\r\n" for character in normalized):
        raise ValueError("Probe text contains a disallowed control character")
    if text_type not in {"query", "document"}:
        raise ValueError("Probe text_type is invalid")
    return {
        "text_type": text_type,
        "character_count": len(normalized),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "text_persisted": False,
    }


def current_runtime_card() -> dict[str, Any]:
    from app.core.config import get_settings

    settings = get_settings()
    return {
        "embedding_api_protocol": str(settings.embedding_api_protocol),
        "embedding_model": str(settings.embedding_model),
        "embedding_dimensions": int(settings.embedding_dimensions),
        "embedding_base_url_present": bool(settings.embedding_base_url),
        "embedding_api_key_present": bool(settings.embedding_api_key),
        "model_bridge_enabled": bool(settings.model_bridge_enabled),
        "model_fallback_enabled": bool(settings.enable_model_fallback),
    }


def sync_model_runtime_card() -> dict[str, Any]:
    from _context_graph_maintenance import prepare_runtime_for_model_io

    card = prepare_runtime_for_model_io()
    return {
        key: card.get(key)
        for key in sorted(SAFE_BRIDGE_SYNC_FIELDS)
        if key in card
    }


async def execute_embedding_probe(
    text: str,
    *,
    text_type: str,
    timeout_seconds: float,
):
    from app.services.embeddings import EmbeddingProvider

    provider = EmbeddingProvider()
    return await asyncio.wait_for(
        provider.embed_texts_with_meta([text], text_type=text_type),
        timeout=timeout_seconds,
    )


def _safe_bridge_response_token(value: Any) -> str | None:
    token = str(value or "").strip().casefold()
    return token if SAFE_BRIDGE_RESPONSE_TOKEN.fullmatch(token) else None


async def execute_bridge_probe(
    text: str,
    *,
    text_type: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    import httpx

    from app.core.config import get_settings
    from app.services import runtime_settings

    settings = get_settings()
    base_url = str(settings.embedding_base_url or "").rstrip("/")
    if not settings.model_bridge_enabled or not runtime_settings._bridge_target_is_self(
        base_url,
        settings,
    ):
        raise RuntimeError("Bridge probe requires the exact local model bridge target")
    payload = {
        "model": str(settings.embedding_model),
        "input": [text],
        "encoding_format": "float",
        "dimensions": int(settings.embedding_dimensions),
    }
    headers = {
        "Authorization": f"Bearer {settings.embedding_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(
        timeout=timeout_seconds,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        async with client.stream(
            "POST",
            f"{base_url}/embeddings",
            json=payload,
            headers=headers,
        ) as response:
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_BRIDGE_PROBE_RESPONSE_BYTES:
                    raise RuntimeError("Bridge probe response exceeded the byte bound")
                chunks.append(chunk)
            body = b"".join(chunks)
            content_type = str(response.headers.get("Content-Type") or "")
            content_type_json = content_type.split(";", 1)[0].strip().casefold() in {
                "application/json",
                "application/problem+json",
            }
            data: dict[str, Any] = {}
            if content_type_json:
                try:
                    decoded = json.loads(body.decode("utf-8"))
                    data = decoded if isinstance(decoded, dict) else {}
                except (UnicodeError, json.JSONDecodeError):
                    data = {}
            error = data.get("error") if isinstance(data.get("error"), dict) else {}
            response_data = data.get("data") if isinstance(data.get("data"), list) else []
            dimensions = [
                len(item.get("embedding"))
                for item in response_data
                if isinstance(item, dict) and isinstance(item.get("embedding"), list)
            ]
            return {
                "http_status": int(response.status_code),
                "content_type_json": content_type_json,
                "response_bytes": len(body),
                "error_present": bool(error),
                "error_code": _safe_bridge_response_token(error.get("code")),
                "error_route": _safe_bridge_response_token(error.get("route")),
                "vector_count": len(dimensions),
                "dimensions": dimensions,
                "expected_dimensions": int(settings.embedding_dimensions),
                "dimension_match": dimensions == [int(settings.embedding_dimensions)],
                "response_body_persisted": False,
                "vectors_persisted": False,
                "credentials_persisted": False,
            }


def exception_type_chain(exc: BaseException) -> list[str]:
    chain: list[str] = []
    current: BaseException | None = exc
    seen: set[int] = set()
    for _depth in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        chain.append(type(current).__name__[:128])
        next_error = current.__cause__
        if next_error is None or id(next_error) in seen:
            next_error = current.__context__
        current = next_error
    return chain


def failure_card(exc: BaseException) -> dict[str, Any]:
    try:
        from app.services.error_sanitizer import external_failure_classification

        classification = external_failure_classification(exc)
    except Exception:
        classification = {
            "protocol_version": "external_failure_classification_v1",
            "classified": False,
            "classification_source": "classification_unavailable",
            "outer_error_type": type(exc).__name__[:128],
            "classified_error_type": None,
            "cause_depth": None,
            "service": None,
            "phase": None,
            "http_status": None,
            "error_code": None,
            "retryable": None,
        }
    return {
        "classification": classification,
        "exception_type_chain": exception_type_chain(exc),
        "exception_message_persisted": False,
        "provider_response_persisted": False,
    }


def vector_result_card(result: Any, *, expected_dimensions: int) -> dict[str, Any]:
    vectors = list(result.vectors)
    dimensions = [len(vector) for vector in vectors]
    finite = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for vector in vectors
        for value in vector
    )
    nonzero = all(any(float(value) != 0.0 for value in vector) for vector in vectors)
    norms = [
        math.sqrt(sum(float(value) * float(value) for value in vector))
        for vector in vectors
    ] if finite else []
    return {
        "provider": str(result.provider),
        "external_called": bool(result.external_called),
        "fallback_reason_present": bool(result.fallback_reason),
        "vector_count": len(vectors),
        "dimensions": dimensions,
        "expected_dimensions": expected_dimensions,
        "dimension_match": dimensions == [expected_dimensions],
        "all_values_finite": finite,
        "all_vectors_nonzero": nonzero,
        "l2_norms": [round(value, 6) for value in norms],
        "vectors_persisted": False,
        "provider_response_persisted": False,
    }


async def run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "script": "probe_embedding_provider",
        "mode": "execute" if args.execute else "dry_run",
        "arm": args.arm,
        "execute": bool(args.execute),
        "network_call_count": 0,
        "input": probe_input_card(args.text, text_type=args.text_type),
        "runtime": current_runtime_card(),
        "provider_response_persisted": False,
        "credentials_persisted": False,
        "endpoint_persisted": False,
    }
    timeout_seconds = float(args.timeout_seconds)
    if not math.isfinite(timeout_seconds) or not 1.0 <= timeout_seconds <= 600.0:
        raise ValueError("timeout_seconds must be finite and within 1..600")
    payload["timeout_seconds"] = timeout_seconds
    if not args.execute:
        payload.update(
            {
                "status": "planned",
                "pass": True,
                "impact": "no model bridge sync and no provider network request",
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        return payload, 0
    if payload["runtime"]["model_fallback_enabled"]:
        payload.update(
            {
                "status": "blocked",
                "pass": False,
                "blocking_reason": "model_fallback_must_be_false",
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            }
        )
        return payload, 2
    try:
        payload["model_bridge_sync"] = sync_model_runtime_card()
        payload["network_call_count"] = 1
        if args.arm == "bridge":
            result_card = await execute_bridge_probe(
                args.text,
                text_type=args.text_type,
                timeout_seconds=timeout_seconds,
            )
            passed = bool(
                200 <= result_card["http_status"] < 300
                and result_card["content_type_json"]
                and result_card["vector_count"] == 1
                and result_card["dimension_match"]
                and not result_card["error_present"]
            )
        else:
            result = await execute_embedding_probe(
                args.text,
                text_type=args.text_type,
                timeout_seconds=timeout_seconds,
            )
            result_card = vector_result_card(
                result,
                expected_dimensions=int(payload["runtime"]["embedding_dimensions"]),
            )
            passed = bool(
                result_card["external_called"]
                and not result_card["fallback_reason_present"]
                and result_card["vector_count"] == 1
                and result_card["dimension_match"]
                and result_card["all_values_finite"]
                and result_card["all_vectors_nonzero"]
            )
        payload.update(
            {
                "status": "completed" if passed else "failed",
                "pass": passed,
                "result": result_card,
            }
        )
        exit_code = 0 if passed else 2
    except BaseException as exc:
        payload.update(
            {
                "status": "failed",
                "pass": False,
                "failure": failure_card(exc),
            }
        )
        exit_code = 2
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return payload, exit_code


def main(args: argparse.Namespace | None = None) -> int:
    if args is None:
        args = parse_args()
    try:
        payload, exit_code = asyncio.run(run_probe(args))
    except BaseException as exc:
        payload = {
            "protocol_version": PROBE_PROTOCOL_VERSION,
            "script": "probe_embedding_provider",
            "mode": "execute" if bool(getattr(args, "execute", False)) else "dry_run",
            "arm": str(getattr(args, "arm", "provider")),
            "execute": bool(getattr(args, "execute", False)),
            "status": "failed",
            "pass": False,
            "network_call_count": 0,
            "failure": failure_card(exc),
            "provider_response_persisted": False,
            "credentials_persisted": False,
            "endpoint_persisted": False,
        }
        exit_code = 2
    report = write_report("probe_embedding_provider", payload)
    print(
        json.dumps(
            {
                "output": str(report),
                "mode": payload["mode"],
                "status": payload["status"],
                "pass": payload["pass"],
            },
            ensure_ascii=False,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

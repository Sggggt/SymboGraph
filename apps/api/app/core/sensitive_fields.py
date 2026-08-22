from __future__ import annotations

import re
import unicodedata
from typing import Any


SENSITIVE_FIELD_KEY_PROTOCOL_VERSION = (
    "semantic_sensitive_field_key_segments_v1"
)
SENSITIVE_FIELD_SCAN_MAX_DEPTH = 128
SENSITIVE_FIELD_SCAN_MAX_ITEMS = 1_000_000

_SINGULAR_SEGMENTS = {
    "credentials": "credential",
    "keys": "key",
    "prompts": "prompt",
    "responses": "response",
    "secrets": "secret",
    "tokens": "token",
}
_SAFE_STATUS_TAILS = frozenset(
    {
        "available",
        "configured",
        "count",
        "counts",
        "digest",
        "enabled",
        "exposed",
        "fields",
        "hash",
        "present",
        "protocol",
        "redacted",
        "redaction",
        "required",
        "source",
        "status",
        "version",
    }
)
_SAFE_TOKEN_OPERATIONAL_SEGMENTS = frozenset(
    {
        "accounting",
        "audit",
        "batch",
        "budget",
        "cache",
        "chat",
        "chunk",
        "client",
        "consumed",
        "consumption",
        "creation",
        "concept",
        "content",
        "context",
        "conversation",
        "cost",
        "count",
        "counts",
        "end",
        "estimate",
        "estimated",
        "extraction",
        "fixed",
        "history",
        "hint",
        "input",
        "json",
        "label",
        "length",
        "limit",
        "local",
        "max",
        "maximum",
        "mid",
        "min",
        "minimum",
        "model",
        "mode",
        "neighbor",
        "original",
        "output",
        "overlap",
        "package",
        "per",
        "protocol",
        "quality",
        "question",
        "read",
        "remaining",
        "request",
        "selected",
        "size",
        "span",
        "start",
        "sufficiency",
        "task",
        "tokenizer",
        "total",
        "usage",
        "used",
        "version",
        "window",
        "write",
    }
)
_DANGEROUS_STORAGE_SEGMENTS = frozenset(
    {
        "archive",
        "backup",
        "blob",
        "bundle",
        "copy",
        "snapshot",
        "value",
    }
)
_PUBLIC_PRIVATE_COMPACT_KEYS = frozenset(
    {
        "promptpack",
    }
)
_COMPACT_DANGEROUS_PATTERNS = (
    re.compile(r"(?:^|.*)apikey(?:backup|blob|bundle|copy|secret|value)?$"),
    re.compile(
        r"(?:^|.*)(?:access|admin|api|auth|bearer|client|id|oauth|refresh|"
        r"session|signing)?token(?:archive|backup|blob|bundle|copy|key|secret|"
        r"value)?$"
    ),
    re.compile(r"(?:^|.*)credential(?:archive|backup|blob|bundle|copy|s)?$"),
    re.compile(
        r"(?:^|.*)(?:providerrawresponse|rawproviderresponse|"
        r"providerresponse|providerpayload|rawproviderpayload)"
        r"(?:archive|backup|blob|bundle|copy|snapshot|value)?$"
    ),
    re.compile(
        r"(?:^|.*)system(?:content|prompt)"
        r"(?:archive|backup|blob|bundle|copy|snapshot|value)?$"
    ),
)


def semantic_key_segments(value: object) -> tuple[str, ...]:
    """Return deterministic semantic segments for case/separator variants."""

    text = unicodedata.normalize("NFKC", str(value).strip())
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    raw_segments = re.findall(r"[a-z0-9]+", text.casefold())
    return tuple(_SINGULAR_SEGMENTS.get(item, item) for item in raw_segments)


def _has_contiguous_segments(
    segments: tuple[str, ...],
    expected: tuple[str, ...],
) -> bool:
    width = len(expected)
    return any(
        segments[index : index + width] == expected
        for index in range(0, len(segments) - width + 1)
    )


def _safe_observability_key(segments: tuple[str, ...]) -> bool:
    if not segments:
        return False
    if segments[0] == "has":
        return True
    if any(item in {"expose", "exposed", "exposes"} for item in segments):
        return True
    return segments[-1] in _SAFE_STATUS_TAILS


def _safe_token_operational_key(segments: tuple[str, ...]) -> bool:
    if "token" not in segments or len(segments) <= 1:
        return False
    non_token = tuple(item for item in segments if item != "token")
    return bool(non_token) and all(
        item in _SAFE_TOKEN_OPERATIONAL_SEGMENTS for item in non_token
    )


def sensitive_field_key_reason(
    value: object,
    *,
    include_public_private: bool = False,
) -> str | None:
    """Classify structural secret/provider/prompt field names.

    The protocol deliberately distinguishes credential semantics from ordinary
    token accounting.  For example, ``token_count`` and
    ``context_package_token_budget`` are operational metrics, while
    ``auth_token`` and ``tokenBackup`` are credentials.
    """

    segments = semantic_key_segments(value)
    if not segments:
        return None
    compact = "".join(segments)
    if include_public_private and compact in _PUBLIC_PRIVATE_COMPACT_KEYS:
        return "private_profile_prompt_container"

    has_profile_document = _has_contiguous_segments(
        segments, ("profile", "json")
    ) or compact.startswith("profilejson")
    has_api_key = _has_contiguous_segments(
        segments, ("api", "key")
    ) or "apikey" in compact
    has_provider_response = (
        (
            "provider" in segments
            and (
                "response" in segments
                or "payload" in segments
            )
        )
        or "providerrawresponse" in compact
        or "rawproviderresponse" in compact
        or "providerresponse" in compact
        or "providerpayload" in compact
        or "rawproviderpayload" in compact
    )
    has_system_prompt = (
        "system" in segments
        and (
            "prompt" in segments
            or "content" in segments
        )
    ) or "systemprompt" in compact or "systemcontent" in compact
    has_credential = "credential" in segments or "credential" in compact
    has_authorization = (
        "auth" in segments
        or "authorization" in segments
        or compact in {"auth", "authorization", "authorizationheader"}
        or compact.startswith("authtoken")
    )
    has_secret = (
        "password" in segments
        or "secret" in segments
        or compact.endswith("password")
        or compact.endswith("secret")
    )
    has_token = "token" in segments or any(
        pattern.fullmatch(compact)
        for pattern in _COMPACT_DANGEROUS_PATTERNS[1:2]
    )
    has_dangerous_storage_semantics = bool(
        _DANGEROUS_STORAGE_SEGMENTS.intersection(segments)
    )

    if (
        has_api_key
        or has_provider_response
        or has_credential
        or has_authorization
        or has_secret
        or has_system_prompt
    ) and not has_dangerous_storage_semantics and _safe_observability_key(
        segments
    ):
        return None
    if has_token and _safe_token_operational_key(segments):
        return None

    if has_profile_document:
        return "private_profile_document"
    if has_api_key:
        return "api_key"
    if has_provider_response:
        return "provider_response"
    if has_system_prompt:
        return "system_prompt"
    if has_credential:
        return "credential"
    if has_authorization:
        return "authorization"
    if has_secret:
        return "secret"
    if has_token:
        return "token"
    if any(pattern.fullmatch(compact) for pattern in _COMPACT_DANGEROUS_PATTERNS):
        return "dangerous_compound"
    return None


def _safe_sensitive_observability_value(key: str, value: Any) -> bool:
    """Allow only typed, content-free forms of otherwise sensitive metrics."""

    segments = semantic_key_segments(key)
    if segments == ("provider", "response", "persisted"):
        return value is False
    if segments == ("cacheable", "system", "prompt", "sha256"):
        return bool(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value)
        )
    if segments in {
        ("cacheable", "system", "prompt", "utf8", "byte"),
        ("cacheable", "system", "prompt", "utf8", "bytes"),
    }:
        return type(value) is int and value >= 0
    return False


def sensitive_field_paths(
    value: Any,
    *,
    include_public_private: bool = False,
) -> tuple[str, ...]:
    """Return bounded, deterministic paths without reading or echoing values."""

    paths: list[str] = []
    stack: list[tuple[Any, tuple[str, ...], int]] = [(value, (), 0)]
    visited = 0
    while stack:
        current, path, depth = stack.pop()
        visited += 1
        if visited > SENSITIVE_FIELD_SCAN_MAX_ITEMS:
            raise ValueError("sensitive field scan exceeds its item bound")
        if depth > SENSITIVE_FIELD_SCAN_MAX_DEPTH:
            raise ValueError("sensitive field scan exceeds its depth bound")
        model_dump = getattr(current, "model_dump", None)
        if callable(model_dump):
            current = model_dump()
        if isinstance(current, dict):
            for raw_key, child in current.items():
                key = str(raw_key)
                child_path = (*path, key)
                if sensitive_field_key_reason(
                    key,
                    include_public_private=include_public_private,
                ) and not _safe_sensitive_observability_value(key, child):
                    paths.append(".".join(child_path))
                stack.append((child, child_path, depth + 1))
        elif isinstance(current, (list, tuple)):
            for index, child in enumerate(current):
                stack.append(
                    (child, (*path, f"[{index}]"), depth + 1)
                )
    return tuple(sorted(set(paths)))

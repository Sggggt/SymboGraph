from __future__ import annotations

import re
from typing import Any


SENSITIVE_VALUE_RE = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;\"'}]+|"
    r"(bearer\s+)[^\s,;\"'}]+|"
    r"((?:api[_-]?key|token|secret|password)\s*[=:]\s*[\"']?)[^\s,;\"'}]+|"
    r"((?:api[_-]?key|token|secret|password)\"\s*:\s*\")[^\"]+"
)
SAFE_FAILURE_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
EXTERNAL_FAILURE_CLASSIFICATION_PROTOCOL_VERSION = (
    "external_failure_classification_v1"
)
MAX_EXTERNAL_FAILURE_CAUSE_DEPTH = 8


def sanitize_error_message(message: str | None, *, fallback: str = "External service request failed", max_length: int = 240) -> str:
    text = " ".join(str(message or "").split())
    if not text:
        return fallback

    def replace(match: re.Match[str]) -> str:
        for index in range(1, 8):
            group = match.group(index)
            if group:
                return f"{group}[redacted]"
        return "[redacted]"

    sanitized = SENSITIVE_VALUE_RE.sub(replace, text)
    if len(sanitized) > max_length:
        sanitized = sanitized[: max_length - 1].rstrip() + "..."
    return sanitized or fallback


class ExternalServiceError(RuntimeError):
    def __init__(
        self,
        *,
        service: str,
        phase: str,
        status_code: int | None = None,
        error_code: str | None = None,
        retryable: bool | None = None,
        unsupported_parameters: set[str] | None = None,
    ) -> None:
        self.service = service
        self.phase = phase
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.unsupported_parameters = unsupported_parameters or set()
        super().__init__(self.public_message())

    def public_message(self) -> str:
        parts = [f"{self.service} request failed", f"phase={self.phase}"]
        if self.status_code is not None:
            parts.append(f"http_status={self.status_code}")
        if self.error_code:
            parts.append(f"error_code={sanitize_error_message(self.error_code, fallback='unknown', max_length=80)}")
        if self.retryable is not None:
            parts.append(f"retryable={str(self.retryable).lower()}")
        if self.unsupported_parameters:
            parts.append("unsupported_parameters=" + ",".join(sorted(self.unsupported_parameters)))
        return "; ".join(parts)


def _safe_failure_token(value: Any, *, fallback: str) -> str:
    token = str(value or "").strip().lower()
    return token if SAFE_FAILURE_TOKEN_RE.fullmatch(token) else fallback


def external_failure_classification(exc: BaseException) -> dict[str, Any]:
    """Return bounded scalar evidence without persisting messages or responses."""

    current: BaseException | None = exc
    visited: set[int] = set()
    for depth in range(MAX_EXTERNAL_FAILURE_CAUSE_DEPTH):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        if isinstance(current, ExternalServiceError):
            status = (
                int(current.status_code)
                if type(current.status_code) is int
                and 100 <= int(current.status_code) <= 599
                else None
            )
            retryable = (
                current.retryable if type(current.retryable) is bool else None
            )
            return {
                "protocol_version": (
                    EXTERNAL_FAILURE_CLASSIFICATION_PROTOCOL_VERSION
                ),
                "classified": True,
                "classification_source": "external_service_error",
                "outer_error_type": type(exc).__name__[:128],
                "classified_error_type": type(current).__name__[:128],
                "cause_depth": depth,
                "service": _safe_failure_token(
                    current.service, fallback="noncanonical_service"
                ),
                "phase": _safe_failure_token(
                    current.phase, fallback="noncanonical_phase"
                ),
                "http_status": status,
                "error_code": (
                    _safe_failure_token(
                        current.error_code,
                        fallback="noncanonical_error_code",
                    )
                    if current.error_code is not None
                    else None
                ),
                "retryable": retryable,
            }
        next_error = current.__cause__
        if next_error is None or id(next_error) in visited:
            next_error = current.__context__
        current = next_error
    return {
        "protocol_version": EXTERNAL_FAILURE_CLASSIFICATION_PROTOCOL_VERSION,
        "classified": False,
        "classification_source": "none",
        "outer_error_type": type(exc).__name__[:128],
        "classified_error_type": None,
        "cause_depth": None,
        "service": None,
        "phase": None,
        "http_status": None,
        "error_code": None,
        "retryable": None,
    }


def public_exception_message(exc: Exception, *, fallback: str = "External service request failed") -> str:
    if isinstance(exc, ExternalServiceError):
        return exc.public_message()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return f"{type(exc).__name__}: HTTP {status_code}"
    message = sanitize_error_message(str(exc), fallback=fallback)
    if message == fallback:
        exc_type = type(exc).__name__
        if exc_type and exc_type not in {"Exception", "RuntimeError"}:
            return exc_type
    return message


def external_error_payload(exc: Exception, *, code: str, title: str, message: str, fix_commands: list[str] | None = None) -> dict[str, Any]:
    return {
        "code": code,
        "title": title,
        "message": message,
        "issues": [
            {
                "code": "external_service_error",
                "title": "External service request failed",
                "message": public_exception_message(exc),
                "fix_commands": fix_commands or [],
            }
        ],
    }

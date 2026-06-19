from __future__ import annotations

import re
from typing import Any


SENSITIVE_VALUE_RE = re.compile(
    r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;\"'}]+|"
    r"(bearer\s+)[^\s,;\"'}]+|"
    r"((?:api[_-]?key|token|secret|password)\s*[=:]\s*[\"']?)[^\s,;\"'}]+|"
    r"((?:api[_-]?key|token|secret|password)\"\s*:\s*\")[^\"]+"
)


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


def public_exception_message(exc: Exception, *, fallback: str = "External service request failed") -> str:
    if isinstance(exc, ExternalServiceError):
        return exc.public_message()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return f"{type(exc).__name__}: HTTP {status_code}"
    return sanitize_error_message(str(exc), fallback=fallback)


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

from __future__ import annotations

import hashlib
import asyncio
import http.client
import ipaddress
import json
import logging
import math
import re
import socket
import ssl
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlencode, urlparse

import httpx
from anthropic import AsyncAnthropic

from app.core.concurrency import model_request_slot
from app.core.config import get_settings
from app.services.resource_guard import effective_embedding_batch_size, enforce_memory_budget
from app.services.strategy_profiles import (
    DEFAULT_ANSWER_SYSTEM_PREFIX,
    DEFAULT_ANSWER_LOW_RELEVANCE_CLAUSE,
    DEFAULT_ANSWER_NORMAL_RELEVANCE_CLAUSE,
    DEFAULT_ANSWER_SYSTEM_TEMPLATE,
    DEFAULT_CONTEXT_LABEL,
    DEFAULT_JSON_RESPONSE_FALLBACK_SYSTEM,
    DEFAULT_NO_CONTEXT_EN,
    DEFAULT_NO_CONTEXT_ZH,
    DEFAULT_QUERY_REWRITE_SYSTEM,
    DEFAULT_QUESTION_PERCEPTION_SYSTEM,
    DEFAULT_REFLECTION_REVIEW_SYSTEM,
    ANSWER_GROUNDING_ENVELOPE_PROTOCOL_VERSION,
    active_profile_json,
    compose_immutable_grounded_profile_prompt,
    conversation_preference_prompt_guidance,
    grounded_profile_prompt_protocol_metadata,
    profile_conversation_preferences,
    profile_prompt,
    profile_prompt_template,
    render_prompt_template,
)
from app.services.error_sanitizer import ExternalServiceError, public_exception_message


logger = logging.getLogger(__name__)
MODEL_REQUEST_MAX_ATTEMPTS = 6
MODEL_REQUEST_BACKOFF_CAP_SECONDS = 20.0
MAX_OPENAI_REQUEST_BODY_BYTES = 16 * 1024 * 1024
MAX_OPENAI_RESPONSE_BODY_BYTES = 32 * 1024 * 1024
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION = "anthropic_message_shape_v2"
ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
ANTHROPIC_GROUNDED_ANSWER_MAX_TOKENS = 8192
ANTHROPIC_GROUNDED_ANSWER_OUTPUT_BUDGET_PROTOCOL_VERSION = (
    "anthropic_grounded_answer_claim_bounded_output_budget_v1"
)
PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION = "provider_system_prompt_cache_v1"
ANTHROPIC_SYSTEM_PROMPT_CACHE_CONTROL = {"type": "ephemeral"}
QUESTION_PERCEPTION_JSON_MAX_TOKENS = 1024
MAX_DOH_RESPONSE_BODY_BYTES = 256 * 1024
MAX_DOH_ANSWER_RECORDS = 256
DOH_REQUEST_TIMEOUT_SECONDS = 5.0
DOH_PUBLIC_A_RESOLVERS = (
    ("dns.alidns.com", "223.5.5.5", "/resolve"),
    ("cloudflare-dns.com", "1.1.1.1", "/dns-query"),
)


class FallbackDisabledError(RuntimeError):
    pass


class ProviderJSONShapeError(RuntimeError):
    """A content-free provider JSON parse/type failure eligible for schema repair."""

    def __init__(self, diagnostics: dict[str, Any]) -> None:
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Provider JSON shape validation failed; diagnostics="
            + json.dumps(
                self.diagnostics,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def prefers_chinese_answer(text: str) -> bool:
    from app.services.chinese_text import contains_chinese

    preferred_language = profile_conversation_preferences()["default_language"]
    if preferred_language == "zh":
        return True
    if preferred_language == "en":
        return False
    return contains_chinese(text)


def answer_language_name(text: str) -> str:
    return "Chinese" if prefers_chinese_answer(text) else "English"


def no_context_answer(question: str) -> str:
    profile = active_profile_json()
    if prefers_chinese_answer(question):
        return profile_prompt(profile, "no_context_answer_zh", DEFAULT_NO_CONTEXT_ZH)
    return profile_prompt(profile, "no_context_answer_en", DEFAULT_NO_CONTEXT_EN)


def _format_location_range(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return ""
    start, end = value[0], value[1]
    if start in (None, "") and end in (None, ""):
        return ""
    if start in (None, ""):
        return str(end)
    if end in (None, "") or end == start:
        return str(start)
    return f"{start}-{end}"


def _source_file_name(source_path: str) -> str:
    if not source_path:
        return ""
    parts = [part for part in re.split(r"[\\/]+", source_path.strip()) if part]
    return parts[-1] if parts else source_path.strip()


def _context_source_header(index: int, item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    source_span = metadata.get("source_span") if isinstance(metadata.get("source_span"), dict) else {}
    source_path = str(item.get("source_path") or metadata.get("source_path") or source_span.get("source_path") or "").strip()
    file_name = _source_file_name(source_path)
    document_title = str(item.get("document_title") or metadata.get("document_title") or file_name or "Untitled source").strip()
    section_path = str(metadata.get("section_path") or source_span.get("section_path") or metadata.get("structure_path") or "").strip()
    page_range = _format_location_range(metadata.get("page_range") or source_span.get("page_range"))
    char_span = _format_location_range(metadata.get("char_span") or source_span.get("char_span"))
    partition = str(item.get("partition") or metadata.get("partition") or "").strip()
    lines = [f"[{index}] {document_title}"]
    if file_name:
        lines.append(f"File: {file_name}")
    if source_path:
        lines.append(f"Source path: {source_path}")
    if partition:
        lines.append(f"Partition: {partition}")
    if section_path:
        lines.append(f"Section: {section_path}")
    if page_range:
        lines.append(f"Pages: {page_range}")
    if char_span:
        lines.append(f"Character span: {char_span}")
    return "\n".join(lines)


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def validate_embedding_vectors(vectors: list[list[float]], *, expected_count: int, expected_dimensions: int) -> None:
    if len(vectors) != expected_count:
        raise RuntimeError(f"Embedding response returned {len(vectors)} vector(s), expected {expected_count}")
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimensions:
            raise RuntimeError(f"Embedding vector {index} has dimension {len(vector)}, expected {expected_dimensions}")
        if not all(math.isfinite(float(value)) for value in vector):
            raise RuntimeError(f"Embedding vector {index} contains non-finite values")
        if vector_norm(vector) <= 1e-12:
            raise RuntimeError(f"Embedding vector {index} is all zeros")


def is_degraded_mode() -> bool:
    settings = get_settings()
    return not settings.chat_api_key or not settings.embedding_api_key or not settings.embedding_base_url


def _sync_model_bridge_for_model_io(settings) -> None:
    if not settings.model_bridge_enabled:
        return
    from app.services.runtime_settings import sync_model_bridge_runtime_config

    sync_model_bridge_runtime_config(settings=settings)


def _exception_message(exc: Exception) -> str:
    return public_exception_message(exc)


def _is_unsupported_parameter_error(exc: Exception, parameter_name: str) -> bool:
    if isinstance(exc, ExternalServiceError):
        return parameter_name.lower() in {item.lower() for item in exc.unsupported_parameters}
    message = _exception_message(exc).lower()
    if parameter_name.lower() not in message:
        return False
    return any(
        marker in message
        for marker in (
            "invalidparameter",
            "invalid parameter",
            "unsupported",
            "not support",
            "not supported",
            "unknown parameter",
            "unrecognized",
            "extra inputs",
        )
    )


UNSUPPORTED_PROVIDER_MARKERS = (
    "invalidparameter",
    "invalid parameter",
    "unsupported",
    "not support",
    "not supported",
    "unknown parameter",
    "unrecognized",
    "extra inputs",
)
KNOWN_PROVIDER_PARAMETERS = {
    "dimensions",
    "response_format",
    "json_schema",
    "json_object",
}
KNOWN_PUBLIC_PROVIDER_ERROR_CODES = {
    "invalid_request_error",
    "invalid_parameter",
    "invalidparameter",
    "rate_limit_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "overloaded_error",
    "api_error",
    "upstream_http_error",
    "upstream_redirect_rejected",
}


def _provider_error_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    error = data.get("error")
    return error if isinstance(error, dict) else data


def _provider_error_code(text: str) -> str | None:
    error = _provider_error_json(text)
    for key in ("code", "type", "param"):
        value = error.get(key)
        if isinstance(value, str) and value.strip():
            canonical = value.strip().casefold()
            if canonical in KNOWN_PUBLIC_PROVIDER_ERROR_CODES:
                return canonical
    if error:
        return "provider_error"
    return None


def _detect_unsupported_parameters(text: str) -> set[str]:
    lower = text.lower()
    error = _provider_error_json(text)
    candidates = set(KNOWN_PROVIDER_PARAMETERS)
    if not any(marker in lower for marker in UNSUPPORTED_PROVIDER_MARKERS):
        return set()
    return {parameter for parameter in candidates if parameter.lower() in lower}


def _external_provider_error_from_response(response: httpx.Response, *, phase: str) -> ExternalServiceError:
    body = response.text or ""
    status_code = response.status_code
    return ExternalServiceError(
        service="model_provider",
        phase=phase,
        status_code=status_code,
        error_code=_provider_error_code(body),
        retryable=status_code == 429 or status_code >= 500,
        unsupported_parameters=_detect_unsupported_parameters(body),
    )


def _external_provider_error_from_body(body: str, *, phase: str) -> ExternalServiceError:
    return ExternalServiceError(
        service="model_provider",
        phase=phase,
        error_code=_provider_error_code(body),
        retryable=None,
        unsupported_parameters=_detect_unsupported_parameters(body),
    )


class EmbeddingProvider:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedding_api_protocol = str(
            getattr(self.settings, "embedding_api_protocol", "")
        )
        self._require_supported_embedding_protocol()
        self.embedding_model = str(self.settings.embedding_model)
        self.embedding_dimensions = int(self.settings.embedding_dimensions)

    def _require_supported_embedding_protocol(self) -> None:
        if self.embedding_api_protocol != "openai":
            raise RuntimeError(
                "Unsupported EMBEDDING_API_PROTOCOL; the standard Anthropic "
                "Messages API has no embedding request/response contract"
            )

    def for_embedding_identity(
        self,
        *,
        embedding_api_protocol: Literal["openai"] = "openai",
        embedding_model: str,
        embedding_dimensions: int,
    ) -> "EmbeddingProvider":
        """Bind this provider instance to one frozen vector runtime identity.

        The binding is deliberately instance-local.  Shadow builds must never
        mutate process-global settings while an active KB continues serving a
        different model or dimension.
        """

        if embedding_api_protocol != "openai":
            raise ValueError("embedding_api_protocol must be openai")
        model = str(embedding_model or "").strip()
        dimensions = int(embedding_dimensions)
        if not model:
            raise ValueError("embedding_model must not be empty")
        if dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")
        self.embedding_api_protocol = embedding_api_protocol
        self._require_supported_embedding_protocol()
        self.embedding_model = model
        self.embedding_dimensions = dimensions
        return self

    async def embed_texts(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        return (await self.embed_texts_with_meta(texts, text_type=text_type)).vectors

    async def embed_texts_with_meta(self, texts: list[str], text_type: str = "document") -> "EmbeddingCallResult":
        # Keep protocol rejection outside the fallback try/except.  An unknown
        # embedding transport must never degrade into synthetic vectors.
        self._require_supported_embedding_protocol()
        if not texts:
            return EmbeddingCallResult(vectors=[], provider="none", external_called=False, fallback_reason=None)
        if not self.settings.embedding_base_url:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("EMBEDDING_BASE_URL is required because ENABLE_MODEL_FALLBACK is false")
            return EmbeddingCallResult(
                vectors=[self._fake_embedding(text) for text in texts],
                provider="fake",
                external_called=False,
                fallback_reason="missing_embedding_base_url",
            )
        if not self.settings.embedding_api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError("EMBEDDING_API_KEY is required because ENABLE_MODEL_FALLBACK is false")
            return EmbeddingCallResult(
                vectors=[self._fake_embedding(text) for text in texts],
                provider="fake",
                external_called=False,
                fallback_reason="missing_embedding_api_key",
            )
        try:
            _sync_model_bridge_for_model_io(self.settings)
            vectors = await self._openai_compatible_embeddings(texts, text_type=text_type)
            validate_embedding_vectors(
                vectors,
                expected_count=len(texts),
                expected_dimensions=self.embedding_dimensions,
            )
            return EmbeddingCallResult(vectors=vectors, provider="openai_compatible", external_called=True, fallback_reason=None)
        except Exception as exc:
            if not self.settings.enable_model_fallback:
                raise
            return EmbeddingCallResult(
                vectors=[self._fake_embedding(text) for text in texts],
                provider="fake",
                external_called=True,
                fallback_reason=public_exception_message(exc),
            )

    async def _openai_compatible_embeddings(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        self._require_supported_embedding_protocol()
        vectors: list[list[float]] = []
        start = 0
        while start < len(texts):
            batch_size = effective_embedding_batch_size(self.settings.embedding_batch_size)
            enforce_memory_budget("embedding_batch")
            batch = texts[start : start + batch_size]
            vectors.extend(await self._openai_compatible_embeddings_batch(batch, text_type=text_type))
            start += batch_size
        return vectors

    async def _openai_compatible_embeddings_batch(self, texts: list[str], text_type: str = "document") -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.embedding_model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self.embedding_dimensions,
        }
        base_url = self.settings.embedding_base_url.rstrip("/")
        api_key = self.settings.embedding_api_key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            data = await post_openai_compatible_json(
                f"{base_url}/embeddings",
                payload,
                headers,
                timeout=60.0,
                resolve_ip=self.settings.embedding_resolve_ip,
                purpose="embedding",
            )
        except Exception as exc:
            if not _is_unsupported_parameter_error(exc, "dimensions"):
                raise
            payload.pop("dimensions", None)
            data = await post_openai_compatible_json(
                f"{base_url}/embeddings",
                payload,
                headers,
                timeout=60.0,
                resolve_ip=self.settings.embedding_resolve_ip,
                purpose="embedding",
            )
        return [item["embedding"] for item in data["data"]]

    def _fake_embedding(self, text: str) -> list[float]:
        vector = []
        for idx in range(self.embedding_dimensions):
            digest = hashlib.sha256(f"{idx}:{text}".encode("utf-8")).digest()
            value = int.from_bytes(digest[:4], "big") / 2**32
            vector.append((value * 2.0) - 1.0)
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]


@dataclass(frozen=True)
class EmbeddingCallResult:
    vectors: list[list[float]]
    provider: str
    external_called: bool
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ChatCallResult:
    answer: str
    provider: str
    model: str
    external_called: bool
    fallback_reason: str | None = None
    prompt_protocol_version: str | None = None
    prompt_protocol_hash: str | None = None
    grounding_envelope_protocol_version: str | None = None
    grounding_envelope_hash: str | None = None
    profile_hash: str | None = None
    output_token_budget: int | None = None
    output_token_budget_protocol_version: str | None = None
    provider_call_audit: dict[str, Any] | None = None


async def classify_json_with_budget(
    provider: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    fallback: dict[str, Any] | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Apply a component output cap without coupling callers to one adapter.

    Production ``ChatProvider`` exposes ``classify_json_bounded``. Lightweight
    deterministic test adapters and compatibility adapters may expose only the
    original ``classify_json`` method; those adapters make no external model
    call and therefore do not consume a provider completion budget.
    """

    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 256 <= max_tokens <= 32_768
    ):
        raise ValueError("Structured JSON max_tokens must be an integer in 256..32768")
    bounded = getattr(provider, "classify_json_bounded", None)
    if callable(bounded):
        return await bounded(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            max_tokens=max_tokens,
        )
    return await provider.classify_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        fallback=fallback,
    )


class ChatProvider:
    def __init__(self, purpose: Literal["chat", "graph"] = "chat") -> None:
        self.settings = get_settings()
        if purpose not in {"chat", "graph"}:
            raise ValueError(f"Unknown chat provider purpose: {purpose}")
        self.purpose = purpose
        self.api_protocol: Literal["openai", "anthropic"] = (
            self.settings.graph_api_protocol
            if purpose == "graph"
            else self.settings.chat_api_protocol
        )
        self.api_key = self.settings.graph_api_key if purpose == "graph" else self.settings.chat_api_key
        self.base_url = self.settings.graph_base_url if purpose == "graph" else self.settings.chat_base_url
        self.resolve_ip = self.settings.graph_resolve_ip if purpose == "graph" else self.settings.chat_resolve_ip
        self.model = self.settings.graph_model if purpose == "graph" else self.settings.chat_model
        self.api_key_env_name = "GRAPH_API_KEY" if purpose == "graph" else "CHAT_API_KEY"
        self.last_usage_audit: dict[str, Any] = self._empty_usage_audit()
        self.last_prompt_cache_audit: dict[str, Any] = {
            "protocol_version": PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION,
            "api_protocol": self.api_protocol,
            "cache_mode": (
                "anthropic_explicit_ephemeral"
                if self.api_protocol == "anthropic"
                else "openai_compatible_automatic_prefix"
            ),
            "cacheable_system_prompt_present": False,
            "cacheable_system_prompt_sha256": None,
            "provider_response_persisted": False,
        }

    def _empty_usage_audit(self) -> dict[str, Any]:
        return {
            "protocol_version": PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION,
            "api_protocol": self.api_protocol,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "cache_hit": False,
            "cache_write": False,
            "token_accounting_mode": (
                "provider_reported_anthropic_fields_no_cross_field_inference_v1"
                if self.api_protocol == "anthropic"
                else "provider_reported_openai_total_cached_subset_v1"
            ),
            "usage_present": False,
            "provider_response_persisted": False,
        }

    @staticmethod
    def _usage_int(value: Any) -> int | None:
        if type(value) is int and value >= 0:
            return int(value)
        return None

    def _record_prompt_cache_prefix(self, system_prompt: str) -> None:
        encoded = system_prompt.encode("utf-8")
        api_protocol = str(getattr(self, "api_protocol", "anthropic"))
        self.last_prompt_cache_audit = {
            "protocol_version": PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION,
            "api_protocol": api_protocol,
            "cache_mode": (
                "anthropic_explicit_ephemeral"
                if api_protocol == "anthropic"
                else "openai_compatible_automatic_prefix"
            ),
            "cacheable_system_prompt_present": bool(system_prompt),
            "cacheable_system_prompt_sha256": (
                hashlib.sha256(encoded).hexdigest() if system_prompt else None
            ),
            "cacheable_system_prompt_utf8_bytes": len(encoded),
            "provider_response_persisted": False,
        }

    def _record_provider_usage(
        self,
        usage: Any,
        *,
        protocol: Literal["openai", "anthropic"],
    ) -> None:
        if not isinstance(usage, dict):
            self.last_usage_audit = self._empty_usage_audit()
            return
        if protocol == "anthropic":
            input_tokens = self._usage_int(usage.get("input_tokens"))
            output_tokens = self._usage_int(usage.get("output_tokens"))
            cache_creation = self._usage_int(
                usage.get("cache_creation_input_tokens")
            )
            cache_read = self._usage_int(usage.get("cache_read_input_tokens"))
            # Do not infer a total or hit percentage across these fields.
            # Anthropic-compatible gateways do not all expose the same
            # subset/disjoint accounting semantics. Preserve the provider's
            # counters verbatim and use cache_read > 0 only as the hit fact.
            total_tokens = self._usage_int(usage.get("total_tokens"))
            token_accounting_mode = (
                "provider_reported_anthropic_fields_no_cross_field_inference_v1"
            )
        else:
            input_tokens = self._usage_int(usage.get("prompt_tokens"))
            output_tokens = self._usage_int(usage.get("completion_tokens"))
            total_tokens = self._usage_int(usage.get("total_tokens"))
            details = usage.get("prompt_tokens_details")
            cache_read = (
                self._usage_int(details.get("cached_tokens"))
                if isinstance(details, dict)
                else None
            )
            cache_creation = None
            token_accounting_mode = (
                "provider_reported_openai_total_cached_subset_v1"
            )
        self.last_usage_audit = {
            "protocol_version": PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION,
            "api_protocol": protocol,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "cache_hit": bool((cache_read or 0) > 0),
            "cache_write": bool((cache_creation or 0) > 0),
            "token_accounting_mode": token_accounting_mode,
            "usage_present": any(
                value is not None
                for value in (
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_creation,
                    cache_read,
                )
            ),
            "provider_response_persisted": False,
        }

    def provider_call_audit(self) -> dict[str, Any]:
        return {
            "protocol_version": PROVIDER_PROMPT_CACHE_PROTOCOL_VERSION,
            "prompt_cache": dict(self.last_prompt_cache_audit),
            "usage": dict(self.last_usage_audit),
            "provider_response_persisted": False,
        }

    async def answer_question(self, question: str, contexts: list[dict], history: list[dict] | None = None) -> str:
        return (await self.answer_question_with_meta(question, contexts, history)).answer

    async def answer_question_with_meta(
        self,
        question: str,
        contexts: list[dict],
        history: list[dict] | None = None,
        context_quality: str = "normal",
        *,
        max_factual_claims: int | None = None,
    ) -> ChatCallResult:
        if max_factual_claims is not None and (
            isinstance(max_factual_claims, bool)
            or not isinstance(max_factual_claims, int)
            or not 1 <= max_factual_claims <= 100
        ):
            raise ValueError("max_factual_claims must be an integer in 1..100")
        if not contexts:
            return ChatCallResult(
                answer=no_context_answer(question),
                provider="none",
                model=self.model,
                external_called=False,
                fallback_reason="no_contexts",
            )
        if not self.api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError(f"{self.api_key_env_name} is required because ENABLE_MODEL_FALLBACK is false")
            return ChatCallResult(
                answer=self._extractive_answer(question, contexts),
                provider="extractive_fallback",
                model="local_extractive_template",
                external_called=False,
                fallback_reason=f"missing_{self.api_key_env_name.lower()}",
            )
        try:
            prompt_bundle = self._answer_prompt_bundle(
                question,
                context_quality=context_quality,
            )
            output_token_budget = self._grounded_answer_output_token_budget(
                max_factual_claims=max_factual_claims,
            )
            answer = await self._provider_chat(
                question,
                contexts,
                history or [],
                context_quality=context_quality,
                max_factual_claims=max_factual_claims,
                output_token_budget=output_token_budget,
                _prompt_bundle=prompt_bundle,
            )
            prompt_metadata = dict(prompt_bundle["protocol_metadata"])
            return ChatCallResult(
                answer=answer,
                provider=f"{self.api_protocol}_chat",
                model=self.model,
                external_called=True,
                fallback_reason=None,
                prompt_protocol_version=ANSWER_GROUNDING_ENVELOPE_PROTOCOL_VERSION,
                prompt_protocol_hash=prompt_metadata["prompt_protocol_hash"],
                grounding_envelope_protocol_version=prompt_metadata[
                    "protocol_version"
                ],
                grounding_envelope_hash=prompt_metadata["envelope_hash"],
                profile_hash=prompt_metadata["profile_hash"],
                output_token_budget=output_token_budget,
                output_token_budget_protocol_version=(
                    ANTHROPIC_GROUNDED_ANSWER_OUTPUT_BUDGET_PROTOCOL_VERSION
                    if output_token_budget is not None
                    else None
                ),
                provider_call_audit=self.provider_call_audit(),
            )
        except Exception as exc:
            if not self.settings.enable_model_fallback:
                raise
            return ChatCallResult(
                answer=self._extractive_answer(question, contexts),
                provider="extractive_fallback",
                model="local_extractive_template",
                external_called=True,
                fallback_reason=public_exception_message(exc),
            )

    async def classify_json(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any] | None = None,
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not self.api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError(f"{self.api_key_env_name} is required because ENABLE_MODEL_FALLBACK is false")
            if fallback is None:
                raise FallbackDisabledError(f"{self.api_key_env_name} is required (no fallback provided)")
            return fallback
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 256 <= max_tokens <= 32_768
        ):
            raise ValueError("Structured JSON max_tokens must be an integer in 256..32768")
        if self.api_protocol == "anthropic":
            if self.purpose == "graph":
                input_budget = int(
                    getattr(
                        self.settings,
                        "mid_concept_extraction_max_tokens_per_batch",
                        2400,
                    )
                )
                payload["max_tokens"] = min(
                    32_768,
                    max(ANTHROPIC_DEFAULT_MAX_TOKENS, 4 * input_budget),
                )
            else:
                runtime_cap = int(
                    getattr(self.settings, "chat_json_max_tokens", 12_000)
                )
                payload["max_tokens"] = min(
                    runtime_cap,
                    max_tokens if max_tokens is not None else runtime_cap,
                )
            payload["thinking"] = {"type": "disabled"}
        elif max_tokens is not None:
            payload["max_tokens"] = min(
                int(getattr(self.settings, "chat_json_max_tokens", 12_000)),
                max_tokens,
            )
        try:
            return await self._post_chat_json_with_response_format_fallback(payload)
        except Exception as exc:
            if not self.settings.enable_model_fallback:
                raise
            if fallback is None:
                raise
            return fallback

    async def classify_json_bounded(
        self,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any] | None = None,
        *,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Run structured generation under a server-owned component cap.

        The component cap may only tighten the hot-reloadable Runtime cap. It
        never changes provider routing, authentication, fallback, or schema
        validation semantics.
        """

        return await self.classify_json(
            system_prompt,
            user_prompt,
            fallback,
            max_tokens=max_tokens,
        )

    async def rewrite_question(self, question: str, history: list[dict] | None = None) -> str:
        if not self.api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError(f"{self.api_key_env_name} is required because ENABLE_MODEL_FALLBACK is false")
            return question
        history_text = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in (history or [])[-6:])
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": profile_prompt(active_profile_json(), "query_rewrite_system", DEFAULT_QUERY_REWRITE_SYSTEM),
                },
                {"role": "user", "content": f"History:\n{history_text}\n\nQuestion:\n{question}"},
            ],
            "temperature": 0.0,
        }
        try:
            return (await self._post_chat_text(payload)).strip() or question
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            return question

    async def reflect_answer(self, question: str, answer: str, contexts: list[dict]) -> dict[str, Any]:
        if not self.api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError(f"{self.api_key_env_name} is required because ENABLE_MODEL_FALLBACK is false")
            return {"has_issue": False, "issue_type": "none", "suggestion": ""}
        context_text = "\n\n".join(
            f"[{i+1}] {ctx.get('document_title', '')}\n{ctx.get('content', '')[:600]}"
            for i, ctx in enumerate(contexts)
        )
        profile = active_profile_json()
        reflection_domain = profile_prompt(profile, "reflection_domain", "KnowledgeBase knowledge-base assistant")
        citation_domain = profile_prompt(profile, "citation_domain", "KnowledgeBase excerpts")
        system_prompt = profile_prompt_template(
            profile,
            "reflection_review_system",
            DEFAULT_REFLECTION_REVIEW_SYSTEM,
            {"reflection_domain": reflection_domain, "citation_domain": citation_domain},
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            f"{citation_domain}:\n{context_text}\n\n"
            "Check: 1) Does the answer contain claims not found in the excerpts? 2) Is the question fully answered? 3) Are there contradictions between the answer and excerpts?"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "max_tokens": min(
                int(getattr(self.settings, "chat_json_max_tokens", 12_000)),
                QUESTION_PERCEPTION_JSON_MAX_TOKENS,
            ),
        }
        try:
            result = await self._post_chat_json_with_response_format_fallback(payload)
            return {
                "has_issue": bool(result.get("has_issue")),
                "issue_type": str(result.get("issue_type", "none")),
                "suggestion": str(result.get("suggestion", "")),
            }
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            return {"has_issue": False, "issue_type": "none", "suggestion": ""}

    async def perceive_question(self, question: str, history: list[dict] | None = None) -> dict[str, Any]:
        """Perceive user intent, extract entities, and decompose the question.

        Returns a dict with keys:
        - intent: one of definition, comparison, application, procedure, analysis, unknown
        - entities: list of source-grounded concepts found in the question
        - sub_queries: list of sub-questions if multi-hop
        - needs_graph: whether graph search is likely helpful
        - suggested_strategy: one of global_dense, local_graph, hybrid, community
        """
        if not self.api_key:
            if not self.settings.enable_model_fallback:
                raise FallbackDisabledError(f"{self.api_key_env_name} is required because ENABLE_MODEL_FALLBACK is false")
            return {
                "intent": "unknown",
                "entities": [],
                "sub_queries": [question],
                "needs_graph": False,
                "suggested_strategy": "hybrid",
            }
        history_text = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in (history or [])[-4:])
        profile = active_profile_json()
        perception_domain = profile_prompt(profile, "perception_domain", "context-graph-grounded knowledge-base agent")
        entity_label = profile_prompt(profile, "entity_label", "source-grounded concepts")
        system_prompt = profile_prompt_template(
            profile,
            "question_perception_system",
            DEFAULT_QUESTION_PERCEPTION_SYSTEM,
            {"perception_domain": perception_domain, "entity_label": entity_label},
        )
        user_prompt = (
            f"History:\n{history_text}\n\nQuestion:\n{question}\n\n"
            "Analyze this question and output the JSON perception result."
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        default_result = {
            "intent": "unknown",
            "entities": [],
            "sub_queries": [question],
            "needs_graph": False,
            "suggested_strategy": "hybrid",
        }
        try:
            result = await self._post_chat_json_with_response_format_fallback(payload)
            return {
                "intent": str(result.get("intent", "unknown")).lower(),
                "entities": list(result.get("entities", [])),
                "sub_queries": list(result.get("sub_queries", [question])),
                "needs_graph": bool(result.get("needs_graph", False)),
                "suggested_strategy": str(result.get("suggested_strategy", "hybrid")).lower(),
            }
        except Exception:
            if not self.settings.enable_model_fallback:
                raise
            return default_result

    def _answer_prompt_bundle(
        self,
        question: str,
        *,
        context_quality: str,
    ) -> dict[str, Any]:
        target_language = answer_language_name(question)
        profile = active_profile_json()
        context_label = profile_prompt(profile, "context_label", DEFAULT_CONTEXT_LABEL)
        context_label_lower = context_label[:1].lower() + context_label[1:] if context_label else "excerpts"
        coverage_label = profile_prompt(profile, "coverage_label", "KnowledgeBase materials")
        indexed_coverage_label = profile_prompt(profile, "indexed_coverage_label", "indexed KnowledgeBase materials")
        answer_style_guidance = profile_prompt(profile, "answer_system_prefix", DEFAULT_ANSWER_SYSTEM_PREFIX)
        context_quality_template = profile_prompt(
            profile,
            "answer_low_relevance_clause" if context_quality == "low" else "answer_normal_relevance_clause",
            DEFAULT_ANSWER_LOW_RELEVANCE_CLAUSE if context_quality == "low" else DEFAULT_ANSWER_NORMAL_RELEVANCE_CLAUSE,
        )
        context_quality_clause = render_prompt_template(
            context_quality_template,
            {"coverage_label": coverage_label, "indexed_coverage_label": indexed_coverage_label},
        )
        rendered_profile_guidance = profile_prompt_template(
            profile,
            "answer_system_template",
            DEFAULT_ANSWER_SYSTEM_TEMPLATE,
            {
                "answer_system_prefix": answer_style_guidance,
                "context_label_lower": context_label_lower,
                "target_language": target_language,
                "coverage_label": coverage_label,
                "indexed_coverage_label": indexed_coverage_label,
                "context_quality_clause": context_quality_clause,
            },
        )
        rendered_profile_guidance = (
            rendered_profile_guidance.rstrip()
            + "\n"
            + conversation_preference_prompt_guidance(profile)
        )
        protocol_metadata = grounded_profile_prompt_protocol_metadata(
            profile,
            rendered_profile_guidance,
            component="answer",
        )
        return {
            "system_content": compose_immutable_grounded_profile_prompt(
                rendered_profile_guidance,
                component="answer",
            ),
            "target_language": target_language,
            "context_label": context_label,
            "coverage_label": coverage_label,
            "protocol_metadata": protocol_metadata,
        }

    def _grounded_answer_output_token_budget(
        self,
        *,
        max_factual_claims: int | None,
    ) -> int | None:
        if self.api_protocol != "anthropic":
            return None
        runtime_cap = int(
            getattr(self.settings, "chat_json_max_tokens", 12_000)
        )
        requested = (
            ANTHROPIC_GROUNDED_ANSWER_MAX_TOKENS
            if max_factual_claims is not None and max_factual_claims > 1
            else ANTHROPIC_DEFAULT_MAX_TOKENS
        )
        return min(runtime_cap, requested)

    async def _provider_chat(
        self,
        question: str,
        contexts: list[dict],
        history: list[dict],
        context_quality: str = "normal",
        *,
        max_factual_claims: int | None = None,
        output_token_budget: int | None = None,
        _prompt_bundle: dict[str, Any] | None = None,
    ) -> str:
        citations = "\n\n".join(
            f"{_context_source_header(idx + 1, item)}\nContent:\n{item.get('content') or ''}"
            for idx, item in enumerate(contexts)
        )
        prompt_bundle = _prompt_bundle or self._answer_prompt_bundle(
            question,
            context_quality=context_quality,
        )
        target_language = str(prompt_bundle["target_language"])
        context_label = str(prompt_bundle["context_label"])
        coverage_label = str(prompt_bundle["coverage_label"])
        messages = [
            {
                "role": "system",
                "content": str(prompt_bundle["system_content"]),
            },
            *history,
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Required answer language: {target_language}\n\n"
                    f"{context_label}:\n"
                    f"{citations}\n\n"
                    + (
                        "Note: the above excerpts have been assessed as potentially irrelevant. "
                        "Only cite them if they truly support a specific claim in your answer. "
                        f"If none are relevant, answer without citations and note the lack of {coverage_label} coverage.\n\n"
                        if context_quality == "low"
                        else ""
                    )
                    + "Before finalizing, check that every formula is either inline LaTeX or display LaTeX, "
                    "and that variables are not attached to neighboring words."
                    + (
                        "\n\nHard answer-shape limit: use at most "
                        f"{max_factual_claims} complete factual sentences or list items. "
                        "Combine closely related supported facts instead of splitting them into extra claims. "
                        "Do not output private reasoning or chain-of-thought. End immediately after the final allowed claim."
                        if max_factual_claims is not None
                        else ""
                    )
                ),
            },
        ]
        payload = {"model": self.model, "messages": messages, "temperature": 0.2}
        if output_token_budget is not None:
            payload["max_tokens"] = output_token_budget
            payload["thinking"] = {"type": "disabled"}
        return await self._post_chat_text(payload)

    async def _post_chat_text(self, payload: dict[str, Any]) -> str:
        if self.api_protocol == "anthropic":
            anthropic_payload = self._anthropic_messages_payload(payload)
            return await self._post_anthropic_sdk_text(anthropic_payload)
        system_prompt = "\n\n".join(
            str(message.get("content") or "")
            for message in list(payload.get("messages") or [])
            if isinstance(message, dict) and message.get("role") == "system"
        )
        self._record_prompt_cache_prefix(system_prompt)
        if self.purpose == "chat":
            _sync_model_bridge_for_model_io(self.settings)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = await post_openai_compatible_json(
            f"{self.base_url.rstrip('/')}/chat/completions",
            payload,
            headers,
            timeout=float(self.settings.model_request_timeout_seconds),
            resolve_ip=self.resolve_ip,
            purpose=self.purpose,
        )
        self._record_provider_usage(data.get("usage"), protocol="openai")
        return self._normalize_chat_content(data)

    async def _post_anthropic_sdk_text(
        self,
        payload: dict[str, Any],
    ) -> str:
        normalized_resolve_ip = str(self.resolve_ip or "").strip().lower()
        if normalized_resolve_ip not in {"", "none", "null", "__none__"}:
            raise RuntimeError(
                "Development Anthropic SDK transport cannot honor a pinned resolve IP"
            )
        if self.purpose == "chat":
            _sync_model_bridge_for_model_io(self.settings)
        response = None
        for attempt in range(1, MODEL_REQUEST_MAX_ATTEMPTS + 1):
            try:
                async with model_request_slot():
                    async with AsyncAnthropic(
                        auth_token=str(self.api_key or ""),
                        base_url=self.base_url.rstrip("/"),
                        timeout=float(self.settings.model_request_timeout_seconds),
                        max_retries=0,
                    ) as client:
                        response = await client.messages.create(**payload)
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                safe_status = (
                    int(status_code)
                    if type(status_code) is int and 100 <= status_code <= 599
                    else None
                )
                retryable = _is_retryable_anthropic_sdk_error(
                    exc,
                    status_code=safe_status,
                )
                external_error = ExternalServiceError(
                    service="anthropic",
                    phase="sdk_messages",
                    status_code=safe_status,
                    error_code=type(exc).__name__.lower()[:80],
                    retryable=retryable,
                )
                if not retryable or attempt >= MODEL_REQUEST_MAX_ATTEMPTS:
                    raise external_error from None
                retry_in = min(
                    float(2 ** (attempt - 1)),
                    MODEL_REQUEST_BACKOFF_CAP_SECONDS,
                )
                logger.warning(
                    "Anthropic SDK request retrying after transient error",
                    extra={
                        "attempt": attempt,
                        "max_attempts": MODEL_REQUEST_MAX_ATTEMPTS,
                        "retry_in_seconds": retry_in,
                        "error_type": type(exc).__name__[:80],
                        "status_code": safe_status,
                    },
                )
                await asyncio.sleep(retry_in)
        if response is None:
            raise RuntimeError("Anthropic SDK request completed without a response")
        response_payload = response.model_dump(mode="json")
        self._record_provider_usage(
            response_payload.get("usage"),
            protocol="anthropic",
        )
        stop_reason = response_payload.get("stop_reason")
        if stop_reason not in {"end_turn", "stop_sequence"}:
            error_code = {
                "max_tokens": "incomplete_max_tokens",
                "refusal": "provider_refusal",
            }.get(stop_reason, "invalid_stop_reason")
            raise ExternalServiceError(
                service="anthropic",
                phase="sdk_messages_completion",
                error_code=error_code,
                retryable=False,
            )
        return self._normalize_anthropic_content(response_payload)

    async def _post_chat_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._parse_json_object(await self._post_chat_text(payload))

    async def _post_chat_json_with_response_format_fallback(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for candidate in self._response_format_candidates(payload.get("response_format")):
            candidate_payload = self._payload_with_response_format(payload, candidate)
            try:
                return await self._post_chat_json(candidate_payload)
            except Exception as exc:
                last_error = exc
                if not self._is_unsupported_response_format_error(exc):
                    raise
        safe_error = public_exception_message(last_error) if last_error else "unknown"
        raise RuntimeError(f"Chat JSON request failed after response_format fallback: {safe_error}")

    def _response_format_candidates(self, response_format: dict[str, Any] | None) -> list[dict[str, Any] | None]:
        if self.api_protocol == "anthropic":
            return [None]
        if not response_format:
            return [None]
        response_type = response_format.get("type")
        if response_type == "json_schema":
            return [response_format, {"type": "json_object"}, None]
        if response_type == "json_object":
            return [response_format, None]
        return [response_format, None]

    def _payload_with_response_format(self, payload: dict[str, Any], response_format: dict[str, Any] | None) -> dict[str, Any]:
        candidate = dict(payload)
        if response_format is None:
            candidate.pop("response_format", None)
            json_fallback_system = profile_prompt(active_profile_json(), "json_response_fallback_system", DEFAULT_JSON_RESPONSE_FALLBACK_SYSTEM)
            messages = [dict(item) for item in candidate.get("messages", [])]
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0].get('content', '')}\n{json_fallback_system}"
            else:
                messages.insert(0, {"role": "system", "content": json_fallback_system})
            candidate["messages"] = messages
        else:
            candidate["response_format"] = response_format
        return candidate

    def _normalize_chat_content(self, data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Chat response did not contain choices")
        choice = choices[0]
        if isinstance(choice.get("text"), str):
            return choice["text"]
        message = choice.get("message") or {}
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise RuntimeError("Chat response contained a refusal")
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                return content
            raise RuntimeError("Chat response content is empty")
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(text, dict) and isinstance(text.get("value"), str):
                        parts.append(text["value"])
                    elif isinstance(part.get("content"), str):
                        parts.append(part["content"])
            normalized = "".join(parts).strip()
            if normalized:
                return normalized
        raise RuntimeError("Chat response did not contain text content")

    def _anthropic_messages_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Anthropic request must be one JSON object")
        allowed = {"model", "messages", "temperature", "max_tokens", "thinking"}
        unsupported = set(payload) - allowed - {"response_format"}
        if unsupported:
            raise ValueError("Anthropic request contained unsupported fields")
        model = payload.get("model")
        messages = payload.get("messages")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Anthropic request model must not be empty")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Anthropic request messages must not be empty")
        system_parts: list[str] = []
        converted: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("Anthropic message must contain only role and content")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                raise ValueError("Anthropic message role is unsupported")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Anthropic message content must be non-empty text")
            if role == "system":
                system_parts.append(content)
            else:
                converted.append({"role": role, "content": content})
        if not converted or converted[0]["role"] != "user":
            raise ValueError("Anthropic conversation must begin with a user message")
        max_tokens = payload.get("max_tokens", ANTHROPIC_DEFAULT_MAX_TOKENS)
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32_768:
            raise ValueError("Anthropic max_tokens must be an integer in 1..32768")
        result: dict[str, Any] = {
            "model": model.strip(),
            "max_tokens": max_tokens,
            "messages": converted,
        }
        if system_parts:
            stable_system_prompt = "\n\n".join(system_parts)
            self._record_prompt_cache_prefix(stable_system_prompt)
            # Anthropic's official SDK accepts a system content-block list.
            # The explicit breakpoint keeps the stable Profile + immutable
            # contract prefix cacheable while request data stays in user
            # messages. No prompt text is copied into diagnostics.
            result["system"] = [
                {
                    "type": "text",
                    "text": stable_system_prompt,
                    "cache_control": dict(
                        ANTHROPIC_SYSTEM_PROMPT_CACHE_CONTROL
                    ),
                }
            ]
        else:
            self._record_prompt_cache_prefix("")
        thinking = payload.get("thinking")
        if thinking is not None:
            if thinking != {"type": "disabled"}:
                raise ValueError(
                    "Anthropic structured graph requests require disabled thinking"
                )
            result["thinking"] = {"type": "disabled"}
        temperature = payload.get("temperature")
        if temperature is not None:
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(float(temperature))
                or not 0.0 <= float(temperature) <= 1.0
            ):
                raise ValueError("Anthropic temperature must be finite in 0..1")
            result["temperature"] = float(temperature)
        return result

    def _normalize_anthropic_content(self, data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            raise ProviderJSONShapeError(
                {
                    "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                    "error_code": "response_not_object",
                    "field_path": "$",
                    "json_type": type(data).__name__,
                }
            )
        if "error" in data:
            raise ProviderJSONShapeError(
                {
                    "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                    "error_code": "error_envelope_rejected",
                    "field_path": "$",
                }
            )
        if data.get("type") != "message" or data.get("role") != "assistant":
            raise ProviderJSONShapeError(
                {
                    "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                    "error_code": "assistant_message_required",
                    "field_path": "$",
                }
            )
        content = data.get("content")
        if not isinstance(content, list) or not content or len(content) > 1024:
            raise ProviderJSONShapeError(
                {
                    "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                    "error_code": "bounded_content_blocks_required",
                    "field_path": "content",
                    "json_type": type(content).__name__,
                }
            )
        parts: list[str] = []
        for index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ProviderJSONShapeError(
                    {
                        "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                        "error_code": "content_block_not_object",
                        "field_path": f"content[{index}]",
                        "json_type": type(block).__name__,
                    }
                )
            block_type = block.get("type")
            if block_type == "thinking":
                thinking = block.get("thinking")
                if not isinstance(thinking, str):
                    raise ProviderJSONShapeError(
                        {
                            "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                            "error_code": "thinking_value_required",
                            "field_path": f"content[{index}].thinking",
                            "json_type": type(thinking).__name__,
                        }
                    )
                continue
            if block_type != "text":
                raise ProviderJSONShapeError(
                    {
                        "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                        "error_code": "text_block_required",
                        "field_path": f"content[{index}].type",
                    }
                )
            text = block.get("text")
            if not isinstance(text, str):
                raise ProviderJSONShapeError(
                    {
                        "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                        "error_code": "text_value_required",
                        "field_path": f"content[{index}].text",
                        "json_type": type(text).__name__,
                    }
                )
            parts.append(text)
        normalized = "".join(parts).strip()
        if not normalized:
            raise ProviderJSONShapeError(
                {
                    "protocol_version": ANTHROPIC_MESSAGE_SHAPE_PROTOCOL_VERSION,
                    "error_code": "empty_text_content",
                    "field_path": "content[].text",
                }
            )
        return normalized

    def _is_unsupported_structured_output_error(self, exc: Exception) -> bool:
        message = _exception_message(exc).lower()
        return "json_schema" in message and any(
            marker in message
            for marker in (
                "response_format",
                "invalid",
                "unsupported",
                "not support",
                "not supported",
                "invalidparameter",
                "invalid parameter",
            )
        )

    def _is_unsupported_response_format_error(self, exc: Exception) -> bool:
        if isinstance(exc, ExternalServiceError):
            params = {item.lower() for item in exc.unsupported_parameters}
            return bool(params & {"response_format", "json_schema", "json_object"})
        message = _exception_message(exc).lower()
        if "response_format" not in message and "json_schema" not in message and "json_object" not in message:
            return False
        return any(
            marker in message
            for marker in (
                "invalid",
                "unsupported",
                "not support",
                "not supported",
                "unknown parameter",
                "unrecognized",
                "invalidparameter",
                "invalid parameter",
                "extra inputs",
            )
        )

    def _extractive_answer(self, question: str, contexts: list[dict]) -> str:
        lead = next((item for item in contexts if item.get("metadata", {}).get("content_kind") != "code"), contexts[0])
        profile = active_profile_json()
        strong_source_label = profile_prompt(profile, "strongest_source_label", "knowledge-base source")
        strong_source_label_zh = profile_prompt(profile, "strongest_source_label_zh", "source")
        relevant_section_label = profile_prompt(profile, "relevant_section_label", "the relevant section")
        relevant_section_label_zh = profile_prompt(profile, "relevant_section_label_zh", "section")
        if prefers_chinese_answer(question):
            lines = [
                f"{strong_source_label_zh}: {lead['document_title']} / {lead.get('partition') or relevant_section_label_zh}.",
                lead["snippet"],
            ]
            if len(contexts) > 1:
                lines.append("Other retrieved excerpts provide related background; use the citations to inspect the source material.")
            lines.append(f"Question: {question}")
            return "\n".join(lines)
        lines = [
            f"The strongest {strong_source_label} is {lead['document_title']} in {lead.get('partition') or relevant_section_label}.",
            lead["snippet"],
        ]
        if len(contexts) > 1:
            lines.append("Other retrieved excerpts provide related background; use the citations to inspect the source material.")
        lines.append(f"Question: {question}")
        return "\n".join(lines)

    def _parse_json_object(self, text: str) -> dict[str, Any]:
        text = text.strip()
        raw_bytes = text.encode("utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            shape_card = {
                "protocol_version": "provider_json_text_shape_v1",
                "error_code": "json_decode_error",
                "field_path": "$",
                "utf8_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "starts_with_object": text.startswith("{"),
                "ends_with_object": text.endswith("}"),
                "contains_code_fence": "```" in text,
                "decode_error_position": max(0, int(exc.pos)),
            }
            raise ProviderJSONShapeError(shape_card) from None
        if not isinstance(parsed, dict):
            shape_card = {
                "protocol_version": "provider_json_text_shape_v1",
                "error_code": "json_root_not_object",
                "field_path": "$",
                "utf8_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "json_type": type(parsed).__name__,
            }
            raise ProviderJSONShapeError(shape_card)
        return parsed


async def post_openai_compatible_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    resolve_ip: str | None = None,
    purpose: Literal["chat", "embedding", "graph"],
) -> dict[str, Any]:
    if purpose not in {"chat", "embedding", "graph"}:
        raise ValueError("OpenAI-compatible transport purpose is invalid")
    request_timeout = float(timeout)
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("OpenAI-compatible request timeout must be finite and positive")
    _validated_openai_request_body(payload)
    restricted_headers = _restricted_openai_headers(headers)
    direct_provider_headers = {
        "Authorization": restricted_headers["Authorization"],
        "Content-Type": restricted_headers["Content-Type"],
    }
    normalized_resolve_ip = (resolve_ip or "").strip()
    if normalized_resolve_ip.lower() in {"", "none", "null", "__none__"}:
        normalized_resolve_ip = ""
    from app.services import runtime_settings

    settings = get_settings()
    parsed_url = urlparse(url)
    expected_route = (
        "/embeddings" if purpose == "embedding" else "/chat/completions"
    )
    if not (
        parsed_url.path == expected_route
        or parsed_url.path.endswith("/" + expected_route.lstrip("/"))
    ):
        raise ValueError(
            "OpenAI-compatible URL route does not match its typed purpose"
        )
    local_bridge_target = bool(
        settings.model_bridge_enabled
        and parsed_url.scheme == "http"
        and parsed_url.path in {"/chat/completions", "/embeddings"}
        and not parsed_url.params
        and not parsed_url.query
        and not parsed_url.fragment
        and parsed_url.username is None
        and parsed_url.password is None
        and runtime_settings._bridge_target_is_self(url, settings)
    )
    local_bridge_request = bool(
        local_bridge_target
        and purpose in {"chat", "embedding"}
        and parsed_url.path == expected_route
    )
    if settings.model_bridge_enabled and purpose in {"chat", "embedding"}:
        if not local_bridge_request:
            raise RuntimeError(
                f"Model bridge is enabled; {purpose} transport must use its "
                "exact local bridge route"
            )
    elif purpose == "graph" and local_bridge_target:
        raise RuntimeError(
            "Graph transport must use its separately configured direct HTTPS provider"
        )
    if local_bridge_request:
        validated_url = url
    else:
        validated_url = runtime_settings._validated_bridge_upstream_url(url)
        if normalized_resolve_ip:
            normalized_resolve_ip = runtime_settings._validated_bridge_resolve_ip(
                normalized_resolve_ip
            )
    last_error: Exception | None = None
    max_attempts = MODEL_REQUEST_MAX_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        try:
            async with model_request_slot():
                if not local_bridge_request:
                    if not normalized_resolve_ip:
                        normalized_resolve_ip = await asyncio.to_thread(
                            _resolve_public_provider_ip,
                            str(urlparse(validated_url).hostname or ""),
                            int(urlparse(validated_url).port or 443),
                            min(request_timeout, DOH_REQUEST_TIMEOUT_SECONDS),
                        )
                    return await asyncio.to_thread(
                        _post_json_with_pinned_resolve,
                        validated_url,
                        payload,
                        direct_provider_headers,
                        request_timeout,
                        normalized_resolve_ip,
                    )
                return await _post_json_to_local_bridge(
                    validated_url,
                    payload,
                    direct_provider_headers,
                    request_timeout,
                )
        except Exception as exc:
            last_error = exc
            retryable = _is_retryable_openai_error(exc)
            if attempt >= max_attempts or not retryable:
                if retryable and attempt >= max_attempts:
                    raise RuntimeError(
                        f"OpenAI-compatible request failed after {max_attempts} attempts: {public_exception_message(exc)}"
                    ) from exc
                raise
            retry_in = min(float(2 ** (attempt - 1)), MODEL_REQUEST_BACKOFF_CAP_SECONDS)
            logger.warning(
                "OpenAI-compatible request retrying after transient error",
                extra={
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "retry_in_seconds": retry_in,
                    "error": public_exception_message(exc),
                    "url_host": urlparse(url).netloc,
                },
            )
            await asyncio.sleep(retry_in)
    safe_error = public_exception_message(last_error) if last_error else "unknown"
    raise RuntimeError(f"OpenAI-compatible request failed: {safe_error}")


async def post_anthropic_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: float,
    resolve_ip: str | None = None,
    purpose: Literal["chat", "graph"],
) -> dict[str, Any]:
    if purpose not in {"chat", "graph"}:
        raise ValueError("Anthropic transport purpose is invalid")
    request_timeout = float(timeout)
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("Anthropic request timeout must be finite and positive")
    _validated_openai_request_body(payload)
    restricted_headers = _restricted_anthropic_headers(headers)
    direct_provider_headers = {
        "X-Api-Key": restricted_headers["X-Api-Key"],
        "Anthropic-Version": restricted_headers["Anthropic-Version"],
        "Content-Type": restricted_headers["Content-Type"],
    }
    normalized_resolve_ip = (resolve_ip or "").strip()
    if normalized_resolve_ip.lower() in {"", "none", "null", "__none__"}:
        normalized_resolve_ip = ""
    from app.services import runtime_settings

    settings = get_settings()
    parsed_url = urlparse(url)
    expected_route = "/v1/messages"
    if not (
        parsed_url.path == expected_route
        or parsed_url.path.endswith(expected_route)
    ):
        raise ValueError("Anthropic URL route does not match Messages API")
    local_bridge_target = bool(
        settings.model_bridge_enabled
        and parsed_url.scheme == "http"
        and parsed_url.path == expected_route
        and not parsed_url.params
        and not parsed_url.query
        and not parsed_url.fragment
        and parsed_url.username is None
        and parsed_url.password is None
        and runtime_settings._bridge_target_is_self(url, settings)
    )
    if settings.model_bridge_enabled and purpose == "chat":
        if not local_bridge_target:
            raise RuntimeError(
                "Model bridge is enabled; Anthropic chat transport must use its exact local bridge route"
            )
    elif purpose == "graph" and local_bridge_target:
        raise RuntimeError(
            "Graph transport must use its separately configured direct HTTPS provider"
        )
    if local_bridge_target:
        validated_url = url
    else:
        validated_url = runtime_settings._validated_bridge_upstream_url(url)
        if normalized_resolve_ip:
            normalized_resolve_ip = runtime_settings._validated_bridge_resolve_ip(
                normalized_resolve_ip
            )
    last_error: Exception | None = None
    for attempt in range(1, MODEL_REQUEST_MAX_ATTEMPTS + 1):
        try:
            async with model_request_slot():
                if not local_bridge_target:
                    if not normalized_resolve_ip:
                        normalized_resolve_ip = await asyncio.to_thread(
                            _resolve_public_provider_ip,
                            str(urlparse(validated_url).hostname or ""),
                            int(urlparse(validated_url).port or 443),
                            min(request_timeout, DOH_REQUEST_TIMEOUT_SECONDS),
                        )
                    return await asyncio.to_thread(
                        _post_json_with_pinned_resolve,
                        validated_url,
                        payload,
                        direct_provider_headers,
                        request_timeout,
                        normalized_resolve_ip,
                        protocol="anthropic",
                    )
                return await _post_json_to_local_bridge(
                    validated_url,
                    payload,
                    direct_provider_headers,
                    request_timeout,
                    protocol="anthropic",
                )
        except Exception as exc:
            last_error = exc
            retryable = _is_retryable_openai_error(exc)
            if attempt >= MODEL_REQUEST_MAX_ATTEMPTS or not retryable:
                if retryable and attempt >= MODEL_REQUEST_MAX_ATTEMPTS:
                    raise RuntimeError(
                        "Anthropic request failed after bounded retries: "
                        + public_exception_message(exc)
                    ) from exc
                raise
            retry_in = min(
                float(2 ** (attempt - 1)),
                MODEL_REQUEST_BACKOFF_CAP_SECONDS,
            )
            logger.warning(
                "Anthropic request retrying after transient error",
                extra={
                    "attempt": attempt,
                    "max_attempts": MODEL_REQUEST_MAX_ATTEMPTS,
                    "retry_in_seconds": retry_in,
                    "error": public_exception_message(exc),
                    "url_host": urlparse(url).netloc,
                },
            )
            await asyncio.sleep(retry_in)
    safe_error = public_exception_message(last_error) if last_error else "unknown"
    raise RuntimeError(f"Anthropic request failed: {safe_error}")


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, ExternalServiceError):
        return bool(exc.retryable)
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return status_code == 429 or status_code >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(
        exc,
        (TimeoutError, ConnectionError, ssl.SSLError, OSError, http.client.HTTPException),
    ):
        return True
    message = public_exception_message(exc).lower()
    non_retryable_markers = ("invalidparameter", "invalid parameter", "unauthorized", "forbidden", "401", "403")
    if any(marker in message for marker in non_retryable_markers):
        return False
    retryable_markers = (
        "timeout",
        "timed out",
        "operation timed out",
        "failed to connect",
        "could not connect",
        "connection reset",
        "handshake",
        "schannel",
        "ssl/tls",
        "temporarily unavailable",
        "empty reply",
    )
    return any(marker in message for marker in retryable_markers)


def _is_retryable_anthropic_sdk_error(
    exc: Exception,
    *,
    status_code: int | None,
) -> bool:
    if status_code == 429 or (
        status_code is not None and status_code >= 500
    ):
        return True
    current: BaseException | None = exc
    visited: set[int] = set()
    for _depth in range(6):
        if current is None or id(current) in visited:
            break
        visited.add(id(current))
        if isinstance(
            current,
            (
                httpx.TimeoutException,
                httpx.TransportError,
                TimeoutError,
                ConnectionError,
                ssl.SSLError,
                OSError,
                http.client.HTTPException,
            ),
        ):
            return True
        error_type = type(current)
        if (
            error_type.__module__.split(".", 1)[0] == "anthropic"
            and error_type.__name__ in {"APIConnectionError", "APITimeoutError"}
        ):
            return True
        next_error = current.__cause__
        if next_error is None or id(next_error) in visited:
            next_error = current.__context__
        current = next_error
    return False


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        hostname: str,
        connect_ip: str,
        port: int,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> None:
        self._connect_ip = connect_ip
        super().__init__(hostname, port, timeout=timeout, context=context)
        # HTTPConnection.__init__ installs socket.create_connection as an
        # instance attribute named ``_create_connection``.  Bind our pinned
        # factory after ``super()`` so the TCP address cannot silently fall
        # back to provider-hostname system DNS while HTTPSConnection still
        # retains ``self.host`` for SNI and certificate hostname validation.
        self._create_connection = self._create_pinned_connection

    def _create_pinned_connection(self, address, timeout, source_address):
        del address
        return socket.create_connection(
            (self._connect_ip, self.port),
            timeout,
            source_address,
        )


def _validated_openai_request_body(payload: dict[str, Any]) -> bytes:
    if not isinstance(payload, dict):
        raise ValueError("OpenAI-compatible request must be one JSON object")
    try:
        request_body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("OpenAI-compatible request was not valid JSON") from None
    if len(request_body) > MAX_OPENAI_REQUEST_BODY_BYTES:
        raise RuntimeError("OpenAI-compatible request exceeded the hard byte bound")
    return request_body


def _restricted_openai_headers(headers: dict[str, str]) -> dict[str, str]:
    if not isinstance(headers, dict):
        raise ValueError("OpenAI-compatible request headers must be a mapping")
    normalized = {str(key).casefold(): str(value) for key, value in headers.items()}
    if set(normalized) - {"authorization", "content-type"}:
        raise ValueError("OpenAI-compatible request contained an unsupported header")
    authorization = normalized.get("authorization", "")
    if (
        not authorization
        or authorization != authorization.strip()
        or len(authorization.encode("utf-8")) > 16 * 1024
        or any(ord(char) < 32 or ord(char) == 127 for char in authorization)
    ):
        raise RuntimeError("OpenAI-compatible request has an invalid Authorization header")
    content_type = normalized.get("content-type", "application/json")
    if content_type.split(";", 1)[0].strip().casefold() != "application/json":
        raise ValueError("OpenAI-compatible request Content-Type must be application/json")
    return {
        "Authorization": authorization,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }


def _restricted_anthropic_headers(headers: dict[str, str]) -> dict[str, str]:
    if not isinstance(headers, dict):
        raise ValueError("Anthropic request headers must be a mapping")
    normalized = {str(key).casefold(): str(value) for key, value in headers.items()}
    allowed = {"x-api-key", "anthropic-version", "content-type"}
    if set(normalized) - allowed:
        raise ValueError("Anthropic request contained an unsupported header")
    api_key = normalized.get("x-api-key", "")
    if (
        not api_key
        or api_key != api_key.strip()
        or len(api_key.encode("utf-8")) > 16 * 1024
        or any(ord(char) < 32 or ord(char) == 127 for char in api_key)
    ):
        raise RuntimeError("Anthropic request has an invalid API key header")
    anthropic_version = normalized.get("anthropic-version", "")
    if anthropic_version != ANTHROPIC_VERSION:
        raise ValueError("Anthropic request version header is unsupported")
    content_type = normalized.get("content-type", "application/json")
    if content_type.split(";", 1)[0].strip().casefold() != "application/json":
        raise ValueError("Anthropic request Content-Type must be application/json")
    return {
        "X-Api-Key": api_key,
        "Anthropic-Version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }


def _validated_json_response_body(
    response_body: bytes,
    *,
    content_type: str,
    content_encoding: str,
) -> dict[str, Any]:
    if len(response_body) > MAX_OPENAI_RESPONSE_BODY_BYTES:
        raise RuntimeError("OpenAI-compatible response exceeded the hard byte bound")
    header_error = _json_response_header_error_code(
        content_type=content_type,
        content_encoding=content_encoding,
    )
    if header_error == "unsupported_content_encoding":
        raise RuntimeError("OpenAI-compatible response used an unsupported content encoding")
    if header_error == "non_json_content_type":
        raise RuntimeError("OpenAI-compatible response Content-Type was not JSON")
    try:
        response_text = response_body.decode("utf-8")
        data = _strict_json_loads(response_text)
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise RuntimeError("OpenAI-compatible response was not valid JSON") from None
    if not isinstance(data, dict):
        raise RuntimeError("OpenAI-compatible response was not one JSON object")
    return data


def _json_response_header_error_code(
    *,
    content_type: str,
    content_encoding: str,
) -> str | None:
    if content_encoding.strip().casefold() not in {"", "identity"}:
        return "unsupported_content_encoding"
    media_type = content_type.split(";", 1)[0].strip().casefold()
    if media_type != "application/json" and not media_type.endswith("+json"):
        return "non_json_content_type"
    return None


def _strict_json_loads(value: str) -> Any:
    def reject_constant(constant: str) -> None:
        del constant
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = item
        return result

    return json.loads(
        value,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _validated_content_length(
    value: str | None,
    *,
    max_bytes: int = MAX_OPENAI_RESPONSE_BODY_BYTES,
) -> None:
    if value in {None, ""}:
        return
    text = str(value)
    if not text.isascii() or not text.isdecimal():
        raise RuntimeError("OpenAI-compatible response Content-Length was invalid")
    length = int(text, 10)
    if length < 0 or length > max_bytes:
        raise RuntimeError("OpenAI-compatible response exceeded the hard byte bound")


async def _post_json_to_local_bridge(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    *,
    protocol: Literal["openai", "anthropic"] = "openai",
) -> dict[str, Any]:
    request_body = _validated_openai_request_body(payload)
    normalized_headers = {
        str(key).casefold(): str(value) for key, value in headers.items()
    }
    if protocol == "anthropic" and set(normalized_headers) == {
        "x-api-key",
        "anthropic-version",
        "content-type",
        "accept",
        "accept-encoding",
    }:
        if (
            normalized_headers["accept"] != "application/json"
            or normalized_headers["accept-encoding"] != "identity"
        ):
            raise ValueError("Anthropic local bridge headers were not canonical")
        headers = {
            "X-Api-Key": normalized_headers["x-api-key"],
            "Anthropic-Version": normalized_headers["anthropic-version"],
            "Content-Type": normalized_headers["content-type"],
        }
    elif protocol == "openai" and set(normalized_headers) == {
        "authorization",
        "content-type",
        "accept",
        "accept-encoding",
    }:
        if (
            normalized_headers["accept"] != "application/json"
            or normalized_headers["accept-encoding"] != "identity"
        ):
            raise ValueError("OpenAI local bridge headers were not canonical")
        headers = {
            "Authorization": normalized_headers["authorization"],
            "Content-Type": normalized_headers["content-type"],
        }
    restricted_headers = (
        _restricted_anthropic_headers(headers)
        if protocol == "anthropic"
        else _restricted_openai_headers(headers)
    )
    async with httpx.AsyncClient(
        timeout=timeout,
        trust_env=False,
        follow_redirects=False,
    ) as client:
        async with client.stream(
            "POST",
            url,
            content=request_body,
            headers=restricted_headers,
        ) as response:
            if 300 <= response.status_code < 400:
                raise ExternalServiceError(
                    service="model_bridge",
                    phase="local_bridge_redirect",
                    status_code=502,
                    error_code="upstream_redirect_rejected",
                    retryable=False,
                )
            _validated_content_length(response.headers.get("Content-Length"))
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_OPENAI_RESPONSE_BODY_BYTES:
                    raise RuntimeError(
                        "OpenAI-compatible response exceeded the hard byte bound"
                    )
                chunks.append(chunk)
            response_body = b"".join(chunks)
            data = _validated_json_response_body(
                response_body,
                content_type=response.headers.get("Content-Type", ""),
                content_encoding=response.headers.get("Content-Encoding", ""),
            )
            if response.status_code >= 400:
                response_text = json.dumps(data, ensure_ascii=False)
                error_code = _provider_error_code(response_text)
                raise ExternalServiceError(
                    service="model_bridge",
                    phase="http_json",
                    status_code=response.status_code,
                    error_code=error_code,
                    retryable=(
                        False
                        if error_code == "upstream_redirect_rejected"
                        else response.status_code == 429
                        or response.status_code >= 500
                    ),
                    unsupported_parameters=_detect_unsupported_parameters(
                        response_text
                    ),
                )
            if data.get("error"):
                raise _external_provider_error_from_body(
                    json.dumps(data["error"], ensure_ascii=False),
                    phase="http_json",
                )
            return data


def _resolve_public_provider_ip(
    hostname: str,
    port: int,
    timeout: float = DOH_REQUEST_TIMEOUT_SECONDS,
) -> str:
    from app.services.runtime_settings import _is_public_unicast_ip

    del port
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if not _is_public_unicast_ip(str(literal)):
            raise RuntimeError("Provider target is not a public unicast address")
        return str(literal)
    if not hostname or len(hostname.encode("ascii", errors="ignore")) != len(hostname):
        raise RuntimeError("Provider hostname is not canonical ASCII DNS syntax")
    for resolver_hostname, resolver_ip, resolver_path in DOH_PUBLIC_A_RESOLVERS:
        try:
            resolved = _query_doh_public_a(
                resolver_hostname=resolver_hostname,
                resolver_ip=resolver_ip,
                resolver_path=resolver_path,
                provider_hostname=hostname,
                timeout=min(float(timeout), DOH_REQUEST_TIMEOUT_SECONDS),
            )
        except Exception:
            continue
        if resolved and _is_public_unicast_ip(resolved):
            return resolved
    raise RuntimeError("Provider hostname has no verified public A record")


def _query_doh_public_a(
    *,
    resolver_hostname: str,
    resolver_ip: str,
    resolver_path: str,
    provider_hostname: str,
    timeout: float,
) -> str | None:
    from app.services.runtime_settings import _is_public_unicast_ip

    if not _is_public_unicast_ip(resolver_ip):
        raise RuntimeError("DoH resolver IP is not public unicast")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("DoH timeout must be finite and positive")
    query = urlencode({"name": provider_hostname, "type": "A"})
    request_target = f"{resolver_path}?{query}"
    tls_context = ssl.create_default_context()
    if tls_context.verify_mode != ssl.CERT_REQUIRED or not tls_context.check_hostname:
        raise RuntimeError("Verified DoH TLS is unavailable")
    connection = _PinnedHTTPSConnection(
        resolver_hostname,
        resolver_ip,
        443,
        timeout=timeout,
        context=tls_context,
    )
    try:
        connection.request(
            "GET",
            request_target,
            headers={
                "Accept": "application/dns-json",
                "Accept-Encoding": "identity",
            },
        )
        response = connection.getresponse()
        if 300 <= int(response.status) < 400:
            raise RuntimeError("DoH redirect was rejected")
        if int(response.status) != 200:
            raise RuntimeError("DoH resolver returned a non-success status")
        _validated_content_length(
            response.getheader("Content-Length"),
            max_bytes=MAX_DOH_RESPONSE_BODY_BYTES,
        )
        content_encoding = str(response.getheader("Content-Encoding") or "")
        if content_encoding.strip().casefold() not in {"", "identity"}:
            raise RuntimeError("DoH response used an unsupported content encoding")
        media_type = str(response.getheader("Content-Type") or "").split(";", 1)[0].strip().casefold()
        if media_type not in {"application/dns-json", "application/json"}:
            raise RuntimeError("DoH response Content-Type was not DNS JSON")
        response_body = response.read(MAX_DOH_RESPONSE_BODY_BYTES + 1)
        if len(response_body) > MAX_DOH_RESPONSE_BODY_BYTES:
            raise RuntimeError("DoH response exceeded the hard byte bound")
        try:
            data = _strict_json_loads(response_body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError, ValueError):
            raise RuntimeError("DoH response was not valid JSON") from None
        status = data.get("Status") if isinstance(data, dict) else None
        truncated = data.get("TC") if isinstance(data, dict) else None
        if (
            not isinstance(data, dict)
            or type(status) is not int
            or status != 0
            or type(truncated) is not bool
            or truncated
        ):
            raise RuntimeError("DoH response was not a complete successful DNS answer")
        answers = data.get("Answer", [])
        if not isinstance(answers, list) or len(answers) > MAX_DOH_ANSWER_RECORDS:
            raise RuntimeError("DoH answer set exceeded the hard record bound")
        public_a_records: list[str] = []
        for answer in answers:
            if not isinstance(answer, dict):
                raise RuntimeError("DoH answer record was malformed")
            record_type = answer.get("type")
            if (
                type(record_type) is not int
                or not 1 <= record_type <= 65535
            ):
                raise RuntimeError("DoH answer record type was malformed")
            if record_type != 1:
                continue
            value = answer.get("data")
            if not isinstance(value, str) or value != value.strip():
                raise RuntimeError("DoH A record data was malformed")
            try:
                ip_value = ipaddress.ip_address(value)
            except ValueError:
                raise RuntimeError("DoH A record data was malformed") from None
            if (
                not isinstance(ip_value, ipaddress.IPv4Address)
                or str(ip_value) != value
                or not _is_public_unicast_ip(value)
            ):
                raise RuntimeError("DoH returned a non-public A record")
            public_a_records.append(value)
        return sorted(set(public_a_records))[0] if public_a_records else None
    finally:
        connection.close()


def _post_json_with_pinned_resolve(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    resolve_ip: str,
    *,
    protocol: Literal["openai", "anthropic"] = "openai",
) -> dict[str, Any]:
    from app.services import runtime_settings

    normalized_url = runtime_settings._validated_bridge_upstream_url(url)
    normalized_resolve_ip = runtime_settings._validated_bridge_resolve_ip(resolve_ip)
    parsed = urlparse(normalized_url)
    if not parsed.hostname or not normalized_resolve_ip:
        raise ValueError("Invalid pinned OpenAI-compatible target")
    request_body = _validated_openai_request_body(payload)
    restricted_headers = (
        _restricted_anthropic_headers(headers)
        if protocol == "anthropic"
        else _restricted_openai_headers(headers)
    )
    tls_context = ssl.create_default_context()
    if tls_context.verify_mode != ssl.CERT_REQUIRED or not tls_context.check_hostname:
        raise RuntimeError("Verified provider TLS is unavailable")
    port = int(parsed.port or 443)
    request_target = parsed.path or "/"
    connection = _PinnedHTTPSConnection(
        str(parsed.hostname),
        normalized_resolve_ip,
        port,
        timeout=float(timeout),
        context=tls_context,
    )
    try:
        connection.request(
            "POST",
            request_target,
            body=request_body,
            headers=restricted_headers,
        )
        response = connection.getresponse()
        response_status = int(response.status)
        if 300 <= response_status < 400:
            raise ExternalServiceError(
                service="model_provider",
                phase="pinned_https_redirect",
                status_code=502,
                error_code="upstream_redirect_rejected",
                retryable=False,
            )
        _validated_content_length(response.getheader("Content-Length"))
        content_type = str(response.getheader("Content-Type") or "")
        content_encoding = str(response.getheader("Content-Encoding") or "")
        header_error = _json_response_header_error_code(
            content_type=content_type,
            content_encoding=content_encoding,
        )
        if header_error is not None:
            raise ExternalServiceError(
                service="model_provider",
                phase="pinned_https_response_headers",
                status_code=response_status,
                error_code=header_error,
                retryable=response_status == 429 or response_status >= 500,
            )
        response_body = response.read(MAX_OPENAI_RESPONSE_BODY_BYTES + 1)
        data = _validated_json_response_body(
            response_body,
            content_type=content_type,
            content_encoding=content_encoding,
        )
        if response_status >= 400:
            response_text = json.dumps(data, ensure_ascii=False)
            raise ExternalServiceError(
                service="model_provider",
                phase="pinned_https_json",
                status_code=response_status,
                error_code=_provider_error_code(response_text),
                retryable=response_status == 429 or response_status >= 500,
                unsupported_parameters=_detect_unsupported_parameters(
                    response_text
                ),
            )
        if data.get("error"):
            raise _external_provider_error_from_body(
                json.dumps(data["error"], ensure_ascii=False),
                phase="pinned_https_json",
            )
        return data
    finally:
        connection.close()

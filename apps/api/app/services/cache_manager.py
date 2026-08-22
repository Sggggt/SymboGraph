from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.core.config import get_settings


class CacheInvalidationError(RuntimeError):
    """Raised when a required shared-cache invalidation was not delivered."""


ACTIVE_RETRIEVAL_CACHE_WRITE_SESSION_KEY = "active_retrieval_cache_writes_v1"
AGENT_REPLAY_POINTER_WRITE_SESSION_KEY = "agent_replay_pointer_writes_v1"
ORDINARY_QUERY_REPLAY_POINTER_WRITE_SESSION_KEY = (
    "ordinary_query_replay_pointer_writes_v1"
)
ACTIVE_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION = (
    "layered_retrieval_full_identity_key_v3"
)
AGENT_REPLAY_POINTER_KEY_PROTOCOL_VERSION = (
    "agent_provider_free_upstream_pointer_key_v1"
)
ORDINARY_QUERY_REPLAY_POINTER_KEY_PROTOCOL_VERSION = (
    "ordinary_query_provider_free_pointer_key_v1"
)
STRICT_CACHED_JSON_MAX_BYTES = 2 * 1024 * 1024
STRICT_CACHED_JSON_MAX_DEPTH = 128
SENSITIVE_CACHE_KEY_PROTOCOL_VERSION = (
    "recursive_sensitive_cache_key_rejection_v4"
)
AGENT_REPLAY_POINTER_COMPONENT_FIELDS = frozenset(
    {
        "pointer_key_protocol_version",
        "knowledge_base_id",
        "provider_free_retrieval_identity",
        "history_hash",
        "conversation_state_audit_hash",
        "conversation_planner_context_hash",
        "policy_operating_prior_hash",
        "query_provider_protocol_hash",
    }
)
ORDINARY_QUERY_REPLAY_POINTER_COMPONENT_FIELDS = frozenset(
    {
        "pointer_key_protocol_version",
        "knowledge_base_id",
        "provider_free_retrieval_identity",
        "query_provider_protocol_hash",
    }
)
ACTIVE_RETRIEVAL_CACHE_COMPONENT_FIELDS = frozenset(
    {
        "cache_key_protocol_version",
        "knowledge_base_id",
        "query",
        "semantic_entry_query_protocol_version",
        "semantic_entry_query",
        "semantic_entry_query_hash",
        "filters",
        "embedding_text_version",
        "local_hint_protocol_version",
        "contextual_index_hash",
        "contextual_index_business_hash",
        "context_graph_hash",
        "chunk_scope_hash",
        "chunk_business_scope_hash",
        "structure_graph_hash",
        "chunk_relation_graph_hash",
        "relation_rank_score_protocol_hash",
        "relation_raw_strength_protocol_hash",
        "chunk_node_quality_protocol_hash",
        "out_evidence_mass_protocol_hash",
        "in_acceptance_capacity_protocol_hash",
        "relation_quota_protocol_hash",
        "edge_type_calibration_protocol_hash",
        "graph_operating_point_hash",
        "calibration_params_hash",
        "edge_type_calibration_config_hash",
        "rq_membership_hash",
        "rq_prefix_pair_aggregate_hash",
        "rq_membership_protocol_hash",
        "mid_concept_hash",
        "coarse_concept_hash",
        "edge_distance_protocol_hash",
        "edge_projection_protocol_hash",
        "graph_protocol_runtime_identity_hash",
        "traversal_protocol_hash",
        "traversal_observation_budget",
        "query_facet_protocol_hash",
        "query_facets_hash",
        "query_facet_posterior_protocol_hash",
        "query_facet_posterior_enabled",
        "query_facet_posterior_observation_budget",
        "query_facet_posterior_round_budget",
        "query_facet_posterior_convergence_epsilon",
        "runtime_settings_hash",
        "cache_runtime_settings_hash",
        "policy_state_hash",
        "agent_operating_envelope_hash",
        "traversal_operating_envelope_hash",
        "conversation_state_scope_hash",
        "prompt_protocol_hash",
        "profile_hash",
        "canonical_profile_state_hash",
        "canonical_graph_hash_protocol_version",
        "canonical_vector_identity",
        "retrieval_mode",
        "retrieval_granularity",
        "result_top_k",
        "typed_action_executor_protocol_version",
        "typed_action_control_hash",
        "typed_action_entry_targets_hash",
        "typed_action_phase_targets_hash",
        "typed_action_allowed_relation_types",
        "repair_directive_protocol_version",
        "repair_directive_hash",
        "repair_action_type",
    }
)


@dataclass(frozen=True)
class SearchCacheRead:
    status: str
    payload: dict[str, Any] | None
    key_digest: str
    ttl_seconds_remaining: int | None
    poison_reason: str | None = None
    deletion_attempted: bool = False
    deleted: bool = False


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON number is not finite")
    return parsed


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _validate_strict_json_resource_bounds(raw: str) -> None:
    if len(raw.encode("utf-8")) > STRICT_CACHED_JSON_MAX_BYTES:
        raise ValueError("cached JSON exceeds the byte limit")
    depth = 0
    in_string = False
    escaped = False
    for character in raw:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > STRICT_CACHED_JSON_MAX_DEPTH:
                raise ValueError("cached JSON exceeds the nesting limit")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("cached JSON has unbalanced delimiters")


def strict_json_loads(raw: str) -> Any:
    _validate_strict_json_resource_bounds(raw)
    return json.loads(
        raw,
        parse_constant=_reject_nonstandard_json_constant,
        parse_float=_strict_json_float,
        object_pairs_hook=_strict_json_object,
    )


def strict_json_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalized_sensitive_cache_key(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _is_sensitive_cache_key(value: object) -> bool:
    key = _normalized_sensitive_cache_key(value)
    if not key:
        return False
    exact = {
        "api_key",
        "api_secret",
        "api_token",
        "apikey",
        "access_token",
        "admin_token",
        "auth_token",
        "authorization",
        "authorization_header",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "provider_raw_response",
        "provider_response",
        "raw_provider_response",
        "refresh_token",
        "secret",
        "signing_secret",
        "token",
    }
    compact_key = key.replace("_", "")
    compact_sensitive_suffixes = {
        "apikey",
        "authorization",
        "authorizationheader",
        "credential",
        "credentials",
        "password",
        "providerrawresponse",
        "providerresponse",
        "rawproviderresponse",
        "secret",
        "token",
    }
    if key in exact or any(
        compact_key.endswith(suffix)
        for suffix in compact_sensitive_suffixes
    ):
        return True
    suffixes = (
        "_api_key",
        "_authorization",
        "_authorization_header",
        "_credential",
        "_credentials",
        "_password",
        "_provider_raw_response",
        "_provider_response",
        "_raw_provider_response",
        "_access_token",
        "_refresh_token",
        "_admin_token",
        "_auth_token",
        "_bearer_token",
        "_api_token",
        "_client_secret",
        "_api_secret",
        "_signing_secret",
    )
    return key.endswith(suffixes)


def validate_no_sensitive_cache_keys(
    value: Any,
    *,
    field: str,
) -> None:
    """Reject credential/provider-response keys before any Redis write.

    Values are intentionally not inspected or echoed: ordinary query text may
    discuss security terms, while a structural credential key is never an
    admissible cache identity or payload field.
    """

    stack: list[tuple[Any, int]] = [(value, 0)]
    visited = 0
    while stack:
        current, depth = stack.pop()
        visited += 1
        if visited > 1_000_000:
            raise ValueError(
                f"{field} exceeds the bounded sensitive-key scan"
            )
        if depth > STRICT_CACHED_JSON_MAX_DEPTH:
            raise ValueError(
                f"{field} exceeds the bounded sensitive-key scan depth"
            )
        if isinstance(current, dict):
            for raw_key, child in current.items():
                if _is_sensitive_cache_key(raw_key):
                    raise ValueError(
                        f"{field} contains a sensitive credential, token, "
                        "authorization, secret, or provider-response key"
                    )
                stack.append((child, depth + 1))
        elif isinstance(current, (list, tuple)):
            stack.extend((child, depth + 1) for child in current)


def validate_active_retrieval_cache_components(
    knowledge_base_id: str,
    cache_components: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(cache_components, dict):
        raise TypeError("active retrieval cache components must be a JSON object")
    validate_no_sensitive_cache_keys(
        cache_components,
        field="active retrieval cache components",
    )
    actual_fields = frozenset(cache_components)
    if actual_fields != ACTIVE_RETRIEVAL_CACHE_COMPONENT_FIELDS:
        missing = sorted(ACTIVE_RETRIEVAL_CACHE_COMPONENT_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - ACTIVE_RETRIEVAL_CACHE_COMPONENT_FIELDS)
        raise ValueError(
            "active retrieval cache requires the complete versioned identity "
            f"components; missing={missing!r}, unexpected={unexpected!r}"
        )
    if (
        cache_components.get("cache_key_protocol_version")
        != ACTIVE_RETRIEVAL_CACHE_KEY_PROTOCOL_VERSION
    ):
        raise ValueError("active retrieval cache key protocol mismatch")
    if str(cache_components.get("knowledge_base_id") or "") != str(
        knowledge_base_id
    ):
        raise ValueError("active retrieval cache knowledge-base identity mismatch")
    # This is both a serializability gate and a ban on NaN/Infinity or
    # ``default=str`` coercion in an active identity.
    serialized = json.dumps(
        cache_components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return json.loads(serialized)


def validate_agent_replay_pointer_components(
    knowledge_base_id: str,
    pointer_components: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(pointer_components, dict):
        raise TypeError("Agent replay pointer components must be a JSON object")
    validate_no_sensitive_cache_keys(
        pointer_components,
        field="Agent replay pointer components",
    )
    actual_fields = frozenset(pointer_components)
    if actual_fields != AGENT_REPLAY_POINTER_COMPONENT_FIELDS:
        missing = sorted(AGENT_REPLAY_POINTER_COMPONENT_FIELDS - actual_fields)
        unexpected = sorted(
            actual_fields - AGENT_REPLAY_POINTER_COMPONENT_FIELDS
        )
        raise ValueError(
            "Agent replay pointer requires the complete provider-free "
            f"identity; missing={missing!r}, unexpected={unexpected!r}"
        )
    if (
        pointer_components.get("pointer_key_protocol_version")
        != AGENT_REPLAY_POINTER_KEY_PROTOCOL_VERSION
    ):
        raise ValueError("Agent replay pointer key protocol mismatch")
    if str(pointer_components.get("knowledge_base_id") or "") != str(
        knowledge_base_id
    ):
        raise ValueError("Agent replay pointer knowledge-base mismatch")
    validate_active_retrieval_cache_components(
        knowledge_base_id,
        pointer_components.get("provider_free_retrieval_identity"),
    )
    for field in (
        "history_hash",
        "conversation_state_audit_hash",
        "conversation_planner_context_hash",
        "policy_operating_prior_hash",
        "query_provider_protocol_hash",
    ):
        digest = pointer_components.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Agent replay pointer {field} is not SHA-256")
    serialized = json.dumps(
        pointer_components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return json.loads(serialized)


def validate_ordinary_query_replay_pointer_components(
    knowledge_base_id: str,
    pointer_components: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(pointer_components, dict):
        raise TypeError(
            "Ordinary query replay pointer components must be a JSON object"
        )
    validate_no_sensitive_cache_keys(
        pointer_components,
        field="ordinary query replay pointer components",
    )
    actual_fields = frozenset(pointer_components)
    if (
        actual_fields
        != ORDINARY_QUERY_REPLAY_POINTER_COMPONENT_FIELDS
    ):
        missing = sorted(
            ORDINARY_QUERY_REPLAY_POINTER_COMPONENT_FIELDS
            - actual_fields
        )
        unexpected = sorted(
            actual_fields
            - ORDINARY_QUERY_REPLAY_POINTER_COMPONENT_FIELDS
        )
        raise ValueError(
            "Ordinary query replay pointer requires the complete "
            "provider-free identity; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    if (
        pointer_components.get("pointer_key_protocol_version")
        != ORDINARY_QUERY_REPLAY_POINTER_KEY_PROTOCOL_VERSION
    ):
        raise ValueError(
            "Ordinary query replay pointer key protocol mismatch"
        )
    if str(pointer_components.get("knowledge_base_id") or "") != str(
        knowledge_base_id
    ):
        raise ValueError(
            "Ordinary query replay pointer knowledge-base mismatch"
        )
    validate_active_retrieval_cache_components(
        knowledge_base_id,
        pointer_components.get("provider_free_retrieval_identity"),
    )
    digest = pointer_components.get("query_provider_protocol_hash")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(
            character not in "0123456789abcdef"
            for character in digest
        )
    ):
        raise ValueError(
            "Ordinary query replay pointer query_provider_protocol_hash "
            "is not SHA-256"
        )
    serialized = json.dumps(
        pointer_components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return json.loads(serialized)


class CacheManager:
    """Lightweight Redis cache wrapper.

    Cache misses must preserve correctness. When Redis is unavailable this
    manager intentionally disables caching instead of keeping process-local
    state that would diverge across API and worker processes.
    """

    def __init__(self) -> None:
        self._redis = None
        self._settings = get_settings()
        self._try_connect()

    def _try_connect(self) -> None:
        try:
            import redis as redis_lib

            self._redis = redis_lib.from_url(self._settings.redis_url, decode_responses=False, socket_connect_timeout=2, socket_timeout=2)
            self._redis.ping()
        except Exception:
            self._redis = None

    def _key(self, namespace: str, *parts: str) -> str:
        safe = ":".join(str(p) for p in parts)
        return f"kg:{namespace}:{safe}"

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _payload_hash(self, payload: dict[str, Any]) -> str:
        return strict_json_sha256(payload)

    def _search_key(
        self,
        knowledge_base_id: str,
        cache_components: dict[str, Any],
    ) -> tuple[str, str]:
        components = validate_active_retrieval_cache_components(
            knowledge_base_id,
            cache_components,
        )
        digest = self._payload_hash(components)
        return self._key("search", knowledge_base_id, digest), digest

    def _agent_replay_pointer_key(
        self,
        knowledge_base_id: str,
        pointer_components: dict[str, Any],
    ) -> tuple[str, str]:
        components = validate_agent_replay_pointer_components(
            knowledge_base_id,
            pointer_components,
        )
        digest = self._payload_hash(components)
        return (
            self._key("agent_replay", knowledge_base_id, digest),
            digest,
        )

    def _ordinary_query_replay_pointer_key(
        self,
        knowledge_base_id: str,
        pointer_components: dict[str, Any],
    ) -> tuple[str, str]:
        components = (
            validate_ordinary_query_replay_pointer_components(
                knowledge_base_id,
                pointer_components,
            )
        )
        digest = self._payload_hash(components)
        return (
            self._key(
                "ordinary_query_replay",
                knowledge_base_id,
                digest,
            ),
            digest,
        )

    def get_embedding(self, knowledge_base_id: str, query: str, embedding_version: str) -> list[float] | None:
        key = self._key("emb", knowledge_base_id, self._hash(query), embedding_version)
        return self._get(key)

    def set_embedding(self, knowledge_base_id: str, query: str, embedding_version: str, vector: list[float], ttl: int = 600) -> None:
        key = self._key("emb", knowledge_base_id, self._hash(query), embedding_version)
        self._set(key, vector, ttl)

    def get_search_results(
        self,
        knowledge_base_id: str,
        *,
        cache_components: dict[str, Any],
    ) -> dict | None:
        """Read an active retrieval identity envelope.

        Active search has no legacy partial-key branch.  A caller must provide
        the complete request/graph/runtime identity card used by the retrieval
        trace.  The Redis value is JSON only and is never treated as an ORM
        result or evidence.
        """

        read = self.read_search_results(
            knowledge_base_id,
            cache_components=cache_components,
        )
        return read.payload if read.status == "hit" else None

    def read_search_results(
        self,
        knowledge_base_id: str,
        *,
        cache_components: dict[str, Any],
    ) -> SearchCacheRead:
        key, digest = self._search_key(knowledge_base_id, cache_components)
        return self._read_json_object(key, digest)

    def _read_json_object(
        self,
        key: str,
        digest: str,
    ) -> SearchCacheRead:
        if self._redis is None:
            return SearchCacheRead(
                status="unavailable",
                payload=None,
                key_digest=digest,
                ttl_seconds_remaining=None,
            )
        try:
            raw = self._redis.get(key)
        except Exception:
            return SearchCacheRead(
                status="unavailable",
                payload=None,
                key_digest=digest,
                ttl_seconds_remaining=None,
            )
        if raw is None:
            return SearchCacheRead(
                status="miss",
                payload=None,
                key_digest=digest,
                ttl_seconds_remaining=None,
            )
        poison_reason: str | None = None
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if not isinstance(raw, str):
                raise TypeError("Redis value is not UTF-8 JSON")
            payload = strict_json_loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("Redis value is not a JSON object")
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            RecursionError,
            OverflowError,
        ):
            payload = None
            poison_reason = "invalid_json_identity_envelope"

        ttl_seconds_remaining: int | None = None
        if poison_reason is None:
            try:
                ttl_seconds_remaining = int(self._redis.ttl(key))
            except Exception:
                poison_reason = "ttl_unverifiable"
            else:
                if ttl_seconds_remaining == -2:
                    return SearchCacheRead(
                        status="miss",
                        payload=None,
                        key_digest=digest,
                        ttl_seconds_remaining=None,
                    )
                if ttl_seconds_remaining <= 0:
                    poison_reason = "ttl_missing_or_expired"

        if poison_reason is not None:
            deleted = self._delete_key(key)
            return SearchCacheRead(
                status="poison",
                payload=None,
                key_digest=digest,
                ttl_seconds_remaining=ttl_seconds_remaining,
                poison_reason=poison_reason,
                deletion_attempted=True,
                deleted=deleted,
            )
        return SearchCacheRead(
            status="hit",
            payload=payload,
            key_digest=digest,
            ttl_seconds_remaining=ttl_seconds_remaining,
        )

    def read_agent_replay_pointer(
        self,
        knowledge_base_id: str,
        *,
        pointer_components: dict[str, Any],
    ) -> SearchCacheRead:
        key, digest = self._agent_replay_pointer_key(
            knowledge_base_id,
            pointer_components,
        )
        return self._read_json_object(key, digest)

    def read_ordinary_query_replay_pointer(
        self,
        knowledge_base_id: str,
        *,
        pointer_components: dict[str, Any],
    ) -> SearchCacheRead:
        key, digest = self._ordinary_query_replay_pointer_key(
            knowledge_base_id,
            pointer_components,
        )
        return self._read_json_object(key, digest)

    def set_search_results(
        self,
        knowledge_base_id: str,
        payload: dict,
        ttl: int = 300,
        *,
        cache_components: dict[str, Any],
    ) -> None:
        if not isinstance(payload, dict):
            raise TypeError("active retrieval cache payload must be a JSON object")
        validate_no_sensitive_cache_keys(
            payload,
            field="active retrieval cache payload",
        )
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        key, _digest = self._search_key(knowledge_base_id, cache_components)
        self._set(key, payload, ttl)

    def delete_search_results(
        self,
        knowledge_base_id: str,
        *,
        cache_components: dict[str, Any],
    ) -> bool:
        key, _digest = self._search_key(knowledge_base_id, cache_components)
        return self._delete_key(key)

    def set_agent_replay_pointer(
        self,
        knowledge_base_id: str,
        payload: dict[str, Any],
        ttl: int = 300,
        *,
        pointer_components: dict[str, Any],
    ) -> None:
        if not isinstance(payload, dict):
            raise TypeError("Agent replay pointer payload must be a JSON object")
        validate_no_sensitive_cache_keys(
            payload,
            field="Agent replay pointer payload",
        )
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        key, _digest = self._agent_replay_pointer_key(
            knowledge_base_id,
            pointer_components,
        )
        self._set(key, payload, ttl)

    def delete_agent_replay_pointer(
        self,
        knowledge_base_id: str,
        *,
        pointer_components: dict[str, Any],
    ) -> bool:
        key, _digest = self._agent_replay_pointer_key(
            knowledge_base_id,
            pointer_components,
        )
        return self._delete_key(key)

    def set_ordinary_query_replay_pointer(
        self,
        knowledge_base_id: str,
        payload: dict[str, Any],
        ttl: int = 300,
        *,
        pointer_components: dict[str, Any],
    ) -> None:
        if not isinstance(payload, dict):
            raise TypeError(
                "Ordinary query replay pointer payload must be a JSON object"
            )
        validate_no_sensitive_cache_keys(
            payload,
            field="ordinary query replay pointer payload",
        )
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        key, _digest = self._ordinary_query_replay_pointer_key(
            knowledge_base_id,
            pointer_components,
        )
        self._set(key, payload, ttl)

    def delete_ordinary_query_replay_pointer(
        self,
        knowledge_base_id: str,
        *,
        pointer_components: dict[str, Any],
    ) -> bool:
        key, _digest = self._ordinary_query_replay_pointer_key(
            knowledge_base_id,
            pointer_components,
        )
        return self._delete_key(key)

    def _delete_key(self, key: str) -> bool:
        if self._redis is None:
            return False
        try:
            deleted_count = self._redis.delete(key)
        except Exception:
            return False
        return bool(deleted_count)

    @property
    def shared_cache_available(self) -> bool:
        return self._redis is not None

    def get_quality_judgment(self, cache_key: str) -> dict | None:
        value = self._get(self._key("quality_judge", cache_key))
        return value if isinstance(value, dict) else None

    def set_quality_judgment(self, cache_key: str, result: dict, ttl: int = 86400) -> None:
        self._set(self._key("quality_judge", cache_key), result, ttl)

    def get_runtime_state(self, namespace: str, state_id: str) -> dict | None:
        value = self._get(self._key("state", namespace, state_id))
        return value if isinstance(value, dict) else None

    def set_runtime_state(self, namespace: str, state_id: str, state: dict, ttl: int = 86400) -> None:
        self._set(self._key("state", namespace, state_id), state, ttl)

    def delete_runtime_state(self, namespace: str, state_id: str) -> None:
        key = self._key("state", namespace, state_id)
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception:
                pass

    def invalidate_knowledge_base(self, knowledge_base_id: str, *, strict: bool = False) -> bool:
        """Invalidate all shared cache entries scoped to a knowledge base.

        Ordinary cache callers may keep cache-miss-correct best-effort behavior.
        Durable mutation recovery must pass ``strict=True`` so a missing Redis
        client or partial SCAN/DELETE failure remains an observable, retryable
        post-commit side effect instead of being mistaken for completion.
        """

        if self._redis is None and strict:
            self._try_connect()
        if self._redis is None:
            if strict:
                raise CacheInvalidationError(
                    "Shared cache invalidation is unavailable; Redis is not connected"
                )
            return False
        try:
            for key in self._redis.scan_iter(match=f"kg:*:{knowledge_base_id}:*"):
                self._redis.delete(key)
        except Exception as exc:
            if strict:
                raise CacheInvalidationError(
                    "Shared cache invalidation failed before completion"
                ) from exc
            return False
        return True

    def _get(self, key: str) -> Any | None:
        if self._redis:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    if not isinstance(raw, str):
                        return None
                    return json.loads(raw)
            except Exception:
                pass
        return None

    def _set(self, key: str, value: Any, ttl: int) -> None:
        if self._redis:
            try:
                validate_no_sensitive_cache_keys(
                    value,
                    field="Redis cache value",
                )
                payload = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                self._redis.setex(key, ttl, payload.encode("utf-8"))
                return
            except Exception:
                pass


_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def clear_cache_manager() -> None:
    global _cache_manager
    _cache_manager = None


def schedule_search_cache_write_after_commit(
    db: Session,
    *,
    knowledge_base_id: str,
    cache_components: dict[str, Any],
    payload: dict[str, Any],
    ttl: int = 300,
) -> None:
    """Queue a Redis identity-envelope write behind the PostgreSQL commit.

    The queue is session-local and is discarded on rollback.  Redis failure is
    intentionally cache-miss-correct and cannot roll back committed facts.
    """

    cache_components = validate_active_retrieval_cache_components(
        knowledge_base_id,
        cache_components,
    )
    if not isinstance(payload, dict):
        raise TypeError("active retrieval cache payload must be a JSON object")
    validate_no_sensitive_cache_keys(
        payload,
        field="scheduled active retrieval cache payload",
    )
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    frozen_payload = json.loads(serialized_payload)
    cache_key = hashlib.sha256(
        json.dumps(
            cache_components,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    pending = db.info.setdefault(
        ACTIVE_RETRIEVAL_CACHE_WRITE_SESSION_KEY,
        {},
    )
    pending[(str(knowledge_base_id), cache_key)] = {
        "knowledge_base_id": str(knowledge_base_id),
        "cache_components": cache_components,
        "payload": frozen_payload,
        "ttl": max(1, int(ttl)),
    }


def schedule_agent_replay_pointer_write_after_commit(
    db: Session,
    *,
    knowledge_base_id: str,
    pointer_components: dict[str, Any],
    payload: dict[str, Any],
    ttl: int = 300,
) -> None:
    pointer_components = validate_agent_replay_pointer_components(
        knowledge_base_id,
        pointer_components,
    )
    if not isinstance(payload, dict):
        raise TypeError("Agent replay pointer payload must be a JSON object")
    validate_no_sensitive_cache_keys(
        payload,
        field="scheduled Agent replay pointer payload",
    )
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    frozen_payload = json.loads(serialized_payload)
    pointer_key = hashlib.sha256(
        json.dumps(
            pointer_components,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    pending = db.info.setdefault(
        AGENT_REPLAY_POINTER_WRITE_SESSION_KEY,
        {},
    )
    pending[(str(knowledge_base_id), pointer_key)] = {
        "knowledge_base_id": str(knowledge_base_id),
        "pointer_components": pointer_components,
        "payload": frozen_payload,
        "ttl": max(1, int(ttl)),
    }


def schedule_ordinary_query_replay_pointer_write_after_commit(
    db: Session,
    *,
    knowledge_base_id: str,
    pointer_components: dict[str, Any],
    payload: dict[str, Any],
    ttl: int = 300,
) -> None:
    pointer_components = (
        validate_ordinary_query_replay_pointer_components(
            knowledge_base_id,
            pointer_components,
        )
    )
    if not isinstance(payload, dict):
        raise TypeError(
            "Ordinary query replay pointer payload must be a JSON object"
        )
    validate_no_sensitive_cache_keys(
        payload,
        field="scheduled ordinary query replay pointer payload",
    )
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    frozen_payload = json.loads(serialized_payload)
    pointer_key = hashlib.sha256(
        json.dumps(
            pointer_components,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    pending = db.info.setdefault(
        ORDINARY_QUERY_REPLAY_POINTER_WRITE_SESSION_KEY,
        {},
    )
    pending[(str(knowledge_base_id), pointer_key)] = {
        "knowledge_base_id": str(knowledge_base_id),
        "pointer_components": pointer_components,
        "payload": frozen_payload,
        "ttl": max(1, int(ttl)),
    }


def _consume_search_cache_writes(session: Session, *, committed: bool) -> None:
    pending = dict(
        session.info.pop(ACTIVE_RETRIEVAL_CACHE_WRITE_SESSION_KEY, {}) or {}
    )
    if not committed or not pending:
        return
    manager = get_cache_manager()
    setter = getattr(manager, "set_search_results", None)
    if not callable(setter):
        return
    for item in pending.values():
        setter(
            item["knowledge_base_id"],
            item["payload"],
            ttl=item["ttl"],
            cache_components=item["cache_components"],
        )


def _consume_agent_replay_pointer_writes(
    session: Session,
    *,
    committed: bool,
) -> None:
    pending = dict(
        session.info.pop(AGENT_REPLAY_POINTER_WRITE_SESSION_KEY, {}) or {}
    )
    if not committed or not pending:
        return
    manager = get_cache_manager()
    setter = getattr(manager, "set_agent_replay_pointer", None)
    if not callable(setter):
        return
    for item in pending.values():
        setter(
            item["knowledge_base_id"],
            item["payload"],
            ttl=item["ttl"],
            pointer_components=item["pointer_components"],
        )


def _consume_ordinary_query_replay_pointer_writes(
    session: Session,
    *,
    committed: bool,
) -> None:
    pending = dict(
        session.info.pop(
            ORDINARY_QUERY_REPLAY_POINTER_WRITE_SESSION_KEY,
            {},
        )
        or {}
    )
    if not committed or not pending:
        return
    manager = get_cache_manager()
    setter = getattr(
        manager,
        "set_ordinary_query_replay_pointer",
        None,
    )
    if not callable(setter):
        return
    for item in pending.values():
        setter(
            item["knowledge_base_id"],
            item["payload"],
            ttl=item["ttl"],
            pointer_components=item["pointer_components"],
        )


@event.listens_for(Session, "after_commit")
def _active_retrieval_cache_after_commit(session: Session) -> None:
    # SAVEPOINT release is not a durable PostgreSQL fact boundary.
    if session.in_nested_transaction():
        return
    _consume_search_cache_writes(session, committed=True)
    _consume_agent_replay_pointer_writes(session, committed=True)
    _consume_ordinary_query_replay_pointer_writes(
        session,
        committed=True,
    )


@event.listens_for(Session, "after_rollback")
def _active_retrieval_cache_after_rollback(session: Session) -> None:
    # A SAVEPOINT rollback is not a durable write boundary either.  The
    # session-local queue is intentionally fail-safe: discard all pending
    # cache writes instead of trying to infer which entries predated the
    # nested transaction.
    _consume_search_cache_writes(session, committed=False)
    _consume_agent_replay_pointer_writes(session, committed=False)
    _consume_ordinary_query_replay_pointer_writes(
        session,
        committed=False,
    )


@event.listens_for(Session, "after_transaction_end")
def _active_retrieval_cache_after_transaction_end(
    session: Session,
    transaction: Any,
) -> None:
    if transaction.parent is not None:
        return
    # Defensive cleanup for transaction termination paths that did not emit a
    # usable commit/rollback callback.
    session.info.pop(ACTIVE_RETRIEVAL_CACHE_WRITE_SESSION_KEY, None)
    session.info.pop(AGENT_REPLAY_POINTER_WRITE_SESSION_KEY, None)
    session.info.pop(
        ORDINARY_QUERY_REPLAY_POINTER_WRITE_SESSION_KEY,
        None,
    )

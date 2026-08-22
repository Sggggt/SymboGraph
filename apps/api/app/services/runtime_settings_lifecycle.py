from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import (
    EMBEDDING_API_PROTOCOL_ALLOWLIST,
    REBUILD_REQUIRED_SETTINGS,
    Settings,
    get_settings,
    use_runtime_settings_override,
)
from app.models import (
    Chunk,
    ChunkRelationEdge,
    ChunkRelationGraphState,
    ChunkSpan,
    ChunkStructureMapping,
    ChunkVersion,
    CoarseConceptState,
    ContextGraphState,
    Document,
    DocumentVersion,
    KnowledgeBase,
    KnowledgeBaseVectorRuntimeState,
    MidConceptState,
    RuntimeSettingsActivationIntent,
    RuntimeSettingsCandidate,
    RuntimeSettingsShadowBuild,
    VectorShadowBuild,
)
from app.services.chunking import (
    CHUNK_SCHEMA_VERSION,
    CURRENT_EMBEDDING_TEXT_VERSION,
    TOKENIZER_VERSION,
    stable_hash as graph_state_stable_hash,
)
from app.services.context_graph import (
    CHUNK_SCOPE_HASH_PROTOCOL_VERSION,
    active_chunks_query,
    active_graph_admission_gate,
    build_coarse_concept_graph,
    build_mid_concept_graph,
    compute_chunk_scope_hash,
    rebuild_context_graph,
    runtime_settings_state_hash,
    write_context_graph_state,
    write_chunks_and_structure,
)
from app.services.parsers import parse_document
from app.services.storage import (
    freeze_existing_source_snapshot,
    replay_frozen_source_snapshot,
)
from app.services.strategy_profiles import (
    get_active_profile_record,
    use_strategy_profile,
)
from app.services.vector_shadow_lifecycle import (
    LIVE_SHADOW_BUILD_STATUSES,
    REQUIRED_EVALUATION_EVIDENCE,
    REQUIRED_EVALUATION_GATES,
    VECTOR_SHADOW_BUILD_PROTOCOL_VERSION,
    VECTOR_SHADOW_EVALUATION_PROTOCOL_VERSION,
    VectorShadowEvaluation,
    build_vector_shadow_artifacts,
    frozen_vector_schema,
    _set_graph_bundle_state,
    promote_vector_shadow_candidate,
    record_vector_shadow_evaluation,
    resolve_active_vector_runtime_target,
    rollback_vector_shadow_candidate,
    vector_runtime_diagnostics,
    vector_runtime_state_hash,
    vector_schema_hash,
    vector_shadow_evaluation_input_hash,
)


RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION = "runtime_settings_candidate_v2"
RUNTIME_SETTINGS_SHADOW_BUILD_PROTOCOL_VERSION = "runtime_settings_shadow_build_v1"
RUNTIME_SETTINGS_DRY_RUN_PROTOCOL_VERSION = "runtime_settings_candidate_dry_run_v1"
RUNTIME_SETTINGS_EVALUATION_PROTOCOL_VERSION = "runtime_settings_measured_evaluation_v1"
RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION = "runtime_settings_atomic_promotion_v1"
RUNTIME_SETTINGS_ROLLBACK_PROTOCOL_VERSION = "runtime_settings_atomic_rollback_v1"
RUNTIME_SETTINGS_ACTIVATION_PROTOCOL_VERSION = "runtime_settings_activation_intent_v1"
RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION = (
    "runtime_settings_bridge_dual_identity_v2"
)
BRIDGE_IDENTITY_FAILURE_PROTOCOL_VERSION = (
    "runtime_settings_bridge_identity_failure_v1"
)

CHUNK_REBUILD_KEYS = frozenset(
    {"fixed_chunk_size_tokens", "fixed_chunk_overlap_tokens"}
)
VECTOR_REBUILD_KEYS = frozenset(
    {
        "embedding_base_url",
        "embedding_api_protocol",
        "embedding_resolve_ip",
        "embedding_model",
        "embedding_dimensions",
    }
)
CONCEPT_SEMANTIC_NEUTRAL_REBUILD_KEYS = frozenset(
    {
        "mid_concept_extraction_max_model_batches",
        "mid_concept_extraction_max_candidates_per_batch",
        "mid_concept_extraction_max_tokens_per_batch",
        "mid_concept_candidate_keep_threshold",
    }
)
RELATION_OPERATING_POINT_REBUILD_KEYS = frozenset(
    {
        "dense_knn_k_min",
        "dense_knn_k_max",
        "dense_reverse_b_min_base",
        "dense_reverse_b_max_base",
        "dense_reverse_b_min_doc",
        "dense_reverse_b_max_doc",
        "dense_reverse_b_min_lang",
        "dense_reverse_b_max_lang",
        "dense_min_cosine",
        "dense_strong_cosine",
        "cross_doc_out_quota_min",
        "cross_doc_out_quota_max",
        "cross_doc_min_cosine",
        "cross_language_out_quota_min",
        "cross_language_out_quota_max",
        "cross_language_min_cosine",
        "edge_distance_protocol",
        "edge_type_calibration_protocol",
    }
)
MAX_CANDIDATE_KNOWLEDGE_BASES = 64
MAX_SHADOW_DOCUMENTS_PER_KB = 2048
MAX_SHADOW_CHUNKS_PER_KB = 100_000
MAX_STATUS_BUILDS = 128
MAX_ACTIVATION_INTENTS = 256
ACTIVATION_APPLY_LEASE_SECONDS = 300


class RuntimeSettingsBridgeIdentityDriftError(RuntimeError):
    def __init__(
        self,
        *,
        stage: str,
        reason: str,
        expected_identity_hash: str,
        current_identity_hash: str | None,
    ) -> None:
        self.stage = stage
        self.reason = reason
        self.expected_identity_hash = expected_identity_hash
        self.current_identity_hash = current_identity_hash
        super().__init__(
            "Runtime Settings candidate bridge identity failed closed at "
            f"{stage}: {reason}"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash64(value: str | None, *, field_name: str) -> str:
    candidate = str(value or "")
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise RuntimeError(f"{field_name} must be a canonical sha256 digest")
    return candidate


def _settings_value(settings: Settings, key: str) -> Any:
    value = getattr(settings, key)
    if isinstance(value, Path):
        return str(value)
    return value


def rebuild_settings_snapshot(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    return {
        key: _settings_value(settings, key)
        for key in sorted(REBUILD_REQUIRED_SETTINGS)
    }


def _validated_candidate_bridge_identity(
    active: Settings,
    requested: dict[str, Any],
) -> dict[str, Any]:
    """Validate bridge upstream targets without confusing them with serving URLs.

    With the host bridge enabled, ``Settings.chat_base_url`` and
    ``Settings.embedding_base_url`` are the effective API/worker serving
    endpoint.  They are deliberately *not* the provider targets supplied to
    the bridge admin endpoint.  Those upstream values remain in the managed
    runtime env and are resolved by the same helpers used by bridge reload.

    This function is read-only and runs before any candidate DB lookup/write,
    cache invalidation, env publication, bridge call, or model call.
    """

    from app.services import runtime_settings

    env_entries = runtime_settings._env_entries(runtime_settings.ENV_PATH)
    graph_protocol = str(
        requested.get("graph_api_protocol", active.graph_api_protocol) or ""
    )
    graph_target = runtime_settings._validated_provider_target(
        base_url=str(requested.get("graph_base_url", active.graph_base_url) or ""),
        resolve_ip=str(
            requested.get("graph_resolve_ip", active.graph_resolve_ip) or ""
        ),
        protocol=graph_protocol,
        purpose="graph",
    )
    direct_graph_identity = {
        "graph_api_protocol": graph_protocol,
        "graph_target_hash": runtime_settings._hash_bridge_target(
            graph_target["base_url"]
        ),
        "graph_resolve_ip_hash": _stable_hash(graph_target["resolve_ip"]),
    }
    direct_graph_identity_hash = _stable_hash(
        {
            "protocol_version": RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
            "role": "direct_graph_provider_target",
            **direct_graph_identity,
        }
    )
    embedding_requested = bool(
        set(requested)
        & {
            "embedding_base_url",
            "embedding_api_protocol",
            "embedding_resolve_ip",
            "embedding_model",
            "embedding_dimensions",
        }
    )
    if embedding_requested and not active.model_bridge_enabled:
        runtime_settings._validated_provider_target(
            base_url=str(
                requested.get("embedding_base_url", active.embedding_base_url) or ""
            ),
            resolve_ip=str(
                requested.get("embedding_resolve_ip", active.embedding_resolve_ip)
                or ""
            ),
            protocol=str(
                requested.get(
                    "embedding_api_protocol", active.embedding_api_protocol
                )
                or ""
            ),
            purpose="embedding",
        )
    if active.model_bridge_enabled:
        managed_enabled = str(env_entries.get("MODEL_BRIDGE_ENABLED") or "").lower()
        if managed_enabled != "true":
            raise ValueError(
                "Enabled model bridge requires MODEL_BRIDGE_ENABLED=true in the "
                "managed runtime identity"
            )
        try:
            managed_port = int(str(env_entries.get("MODEL_BRIDGE_PORT") or ""))
        except ValueError:
            managed_port = 0
        if managed_port != int(active.model_bridge_port):
            raise ValueError(
                "Managed MODEL_BRIDGE_PORT does not match the effective bridge "
                "serving identity"
            )
    serving_identity = {
        "embedding_api_protocol": active.embedding_api_protocol,
        "chat_base_url_hash": runtime_settings._hash_bridge_target(
            active.chat_base_url
        ),
        "embedding_base_url_hash": runtime_settings._hash_bridge_target(
            active.embedding_base_url
        ),
        "chat_is_bridge_endpoint": runtime_settings._bridge_target_is_self(
            active.chat_base_url, active
        ),
        "embedding_is_bridge_endpoint": runtime_settings._bridge_target_is_self(
            active.embedding_base_url, active
        ),
    }
    serving_identity_hash = _stable_hash(
        {
            "protocol_version": RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
            "role": "effective_serving_endpoint",
            **serving_identity,
        }
    )
    if not active.model_bridge_enabled:
        return {
            "protocol_version": RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
            "bridge_enabled": False,
            "validation": "bypassed_bridge_disabled",
            "serving_identity_hash": serving_identity_hash,
            "base_upstream_config_hash": None,
            "candidate_upstream_config_hash": None,
            "candidate_upstream_changed": False,
            "direct_graph_identity_hash": direct_graph_identity_hash,
            "embedding_api_protocol": active.embedding_api_protocol,
            "provider_response_used_as_fact": False,
            "model_call_count": 0,
        }

    missing_managed = [
        env_key
        for env_key in (
            "CHAT_BASE_URL",
            "EMBEDDING_API_PROTOCOL",
            "EMBEDDING_BASE_URL",
        )
        if env_key not in env_entries
    ]
    if missing_managed:
        raise ValueError(
            "MODEL_BRIDGE_ENABLED=true requires managed upstream provider "
            "URL keys with no process-environment fallback: "
            + ", ".join(missing_managed)
        )
    # Reuse the bridge admin config builder for its field contract and timeout,
    # then replace every upstream field from managed bytes only.  Candidate
    # admission must never inherit a process value that may be the serving URL.
    base_desired = runtime_settings._desired_bridge_config(active, env_entries)
    base_desired["embedding_api_protocol"] = env_entries[
        "EMBEDDING_API_PROTOCOL"
    ]
    for env_key, bridge_key in (
        ("CHAT_BASE_URL", "chat_target_base_url"),
        ("EMBEDDING_BASE_URL", "embedding_target_base_url"),
    ):
        try:
            base_desired[bridge_key] = runtime_settings._validated_bridge_upstream_url(
                env_entries[env_key]
            )
        except ValueError as exc:
            raise ValueError(
                "Managed model bridge upstream HTTPS/public URL validation "
                f"failed for {env_key}: {exc}"
            ) from None
    base_desired["chat_resolve_ip"] = (
        runtime_settings._validated_bridge_resolve_ip(
            env_entries.get("CHAT_RESOLVE_IP", "")
        )
    )
    base_desired["embedding_resolve_ip"] = (
        runtime_settings._validated_bridge_resolve_ip(
            env_entries.get("EMBEDDING_RESOLVE_IP", "")
        )
    )
    if "MODEL_REQUEST_TIMEOUT_SECONDS" in env_entries:
        try:
            managed_timeout = int(env_entries["MODEL_REQUEST_TIMEOUT_SECONDS"])
        except (TypeError, ValueError):
            raise ValueError(
                "Managed MODEL_REQUEST_TIMEOUT_SECONDS is invalid"
            ) from None
        if managed_timeout <= 0:
            raise ValueError(
                "Managed MODEL_REQUEST_TIMEOUT_SECONDS must be positive"
            )
        base_desired["timeout"] = managed_timeout
    candidate_desired = dict(base_desired)
    if "embedding_api_protocol" in requested:
        candidate_desired["embedding_api_protocol"] = requested[
            "embedding_api_protocol"
        ]
    requested_to_bridge_key = {
        # chat_base_url is hot_reloadable and will still be rejected by the
        # candidate field allowlist.  Checking an explicit self-target here
        # first makes the security boundary fail closed with the precise
        # reason rather than treating a bridge loop as an ordinary unknown.
        "chat_base_url": "chat_target_base_url",
        "chat_resolve_ip": "chat_resolve_ip",
        "embedding_base_url": "embedding_target_base_url",
        "embedding_resolve_ip": "embedding_resolve_ip",
    }
    for requested_key, bridge_key in requested_to_bridge_key.items():
        if requested_key not in requested:
            continue
        value = requested[requested_key]
        if bridge_key.endswith("_base_url"):
            try:
                candidate_desired[bridge_key] = (
                    runtime_settings._validated_bridge_upstream_url(
                        str(value or "")
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "Runtime Settings candidate model upstream "
                    f"{requested_key.upper()} is invalid: {exc}"
                ) from None
        else:
            candidate_desired[bridge_key] = (
                runtime_settings._validated_bridge_resolve_ip(str(value or ""))
            )

    self_target_keys = runtime_settings._bridge_self_target_keys(
        candidate_desired, active
    )
    if self_target_keys:
        raise ValueError(
            "Runtime Settings candidate model upstream target points to the "
            "bridge serving endpoint itself: " + ", ".join(self_target_keys)
        )

    base_self_target_keys = runtime_settings._bridge_self_target_keys(
        base_desired, active
    )
    if base_self_target_keys:
        raise ValueError(
            "Managed model bridge upstream target already points to the bridge "
            "serving endpoint itself: " + ", ".join(base_self_target_keys)
        )

    def validate_upstream_urls(desired: dict[str, Any], *, identity_role: str) -> None:
        missing_keys: list[str] = []
        invalid_keys: list[str] = []
        for bridge_key, env_key in (
            ("chat_target_base_url", "CHAT_BASE_URL"),
            ("embedding_target_base_url", "EMBEDDING_BASE_URL"),
        ):
            target = str(desired.get(bridge_key) or "").strip()
            if not target:
                missing_keys.append(env_key)
                continue
            try:
                runtime_settings._validated_bridge_upstream_url(target)
            except ValueError:
                invalid_keys.append(env_key)
        for bridge_key, env_key in (
            ("chat_resolve_ip", "CHAT_RESOLVE_IP"),
            ("embedding_resolve_ip", "EMBEDDING_RESOLVE_IP"),
        ):
            try:
                runtime_settings._validated_bridge_resolve_ip(
                    str(desired.get(bridge_key) or "")
                )
            except ValueError:
                invalid_keys.append(env_key)
        if missing_keys:
            raise ValueError(
                "MODEL_BRIDGE_ENABLED=true requires explicit "
                f"{identity_role} upstream provider targets: "
                + ", ".join(missing_keys)
            )
        if invalid_keys:
            raise ValueError(
                f"Model bridge {identity_role} upstream provider targets must "
                "use HTTPS/public targets and public-unicast resolve IPs: "
                + ", ".join(invalid_keys)
            )

    # A candidate is not a repair path for an already ambiguous/broken active
    # bridge identity.  Both the managed base and the candidate target set must
    # be independently valid before staging can proceed.
    validate_upstream_urls(base_desired, identity_role="managed")
    validate_upstream_urls(candidate_desired, identity_role="candidate")
    base_desired = runtime_settings._validated_desired_bridge_config(base_desired)
    candidate_desired = runtime_settings._validated_desired_bridge_config(
        candidate_desired
    )

    def upstream_identity(desired: dict[str, Any]) -> dict[str, Any]:
        return {
            "chat_api_protocol": str(desired["chat_api_protocol"]),
            "chat_target_hash": runtime_settings._hash_bridge_target(
                str(desired["chat_target_base_url"])
            ),
            "chat_resolve_ip_hash": _stable_hash(
                str(desired.get("chat_resolve_ip") or "")
            ),
            "embedding_target_hash": runtime_settings._hash_bridge_target(
                str(desired["embedding_target_base_url"])
            ),
            "embedding_resolve_ip_hash": _stable_hash(
                str(desired.get("embedding_resolve_ip") or "")
            ),
            "embedding_api_protocol": str(desired["embedding_api_protocol"]),
            "timeout": int(desired["timeout"]),
        }

    base_upstream_identity = upstream_identity(base_desired)
    candidate_upstream_identity = upstream_identity(candidate_desired)
    base_upstream_config_hash = _stable_hash(
        {
            "protocol_version": RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
            "role": "bridge_admin_upstream_provider_target",
            **base_upstream_identity,
        }
    )
    candidate_upstream_config_hash = _stable_hash(
        {
            "protocol_version": RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
            "role": "bridge_admin_upstream_provider_target",
            **candidate_upstream_identity,
        }
    )
    return {
        "protocol_version": RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
        "bridge_enabled": True,
        "validation": "upstream_provider_targets_validated",
        "serving_identity_hash": serving_identity_hash,
        "serving_chat_is_bridge_endpoint": serving_identity[
            "chat_is_bridge_endpoint"
        ],
        "serving_embedding_is_bridge_endpoint": serving_identity[
            "embedding_is_bridge_endpoint"
        ],
        "base_upstream_config_hash": base_upstream_config_hash,
        "candidate_upstream_config_hash": candidate_upstream_config_hash,
        "candidate_upstream_changed": (
            candidate_upstream_config_hash != base_upstream_config_hash
        ),
        "direct_graph_identity_hash": direct_graph_identity_hash,
        "embedding_api_protocol": str(candidate_desired["embedding_api_protocol"]),
        "provider_response_used_as_fact": False,
        "model_call_count": 0,
    }


def _validated_candidate_settings(
    requested_settings: dict[str, Any],
) -> tuple[
    Settings,
    dict[str, Any],
    dict[str, Any],
    list[str],
    dict[str, Any],
]:
    from app.services import runtime_settings

    requested = dict(requested_settings or {})
    if not requested:
        raise ValueError("A rebuild-required candidate must change at least one setting")
    normalized: dict[str, Any] = {}
    for key, value in requested.items():
        if key == "embedding_api_protocol" and (
            type(value) is not str
            or value not in EMBEDDING_API_PROTOCOL_ALLOWLIST
        ):
            raise ValueError(
                "embedding_api_protocol must be an allowlisted embedding API "
                "protocol: "
                + ", ".join(sorted(EMBEDDING_API_PROTOCOL_ALLOWLIST))
            )
        if isinstance(value, str) and key not in {
            "chat_base_url",
            "embedding_base_url",
            "graph_base_url",
        }:
            value = value.strip()
        normalized[key] = value
    active = get_settings()
    bridge_identity = _validated_candidate_bridge_identity(active, normalized)
    unknown = sorted(set(requested).difference(REBUILD_REQUIRED_SETTINGS))
    if unknown:
        raise ValueError(
            "Runtime Settings candidate accepts only rebuild_required fields: "
            + ", ".join(unknown)
        )
    candidate = Settings.model_validate(
        {**active.model_dump(mode="python"), **normalized}
    )
    if candidate.fixed_chunk_overlap_tokens >= candidate.fixed_chunk_size_tokens:
        raise ValueError(
            "fixed_chunk_overlap_tokens must be smaller than fixed_chunk_size_tokens"
        )
    base_snapshot = rebuild_settings_snapshot(active)
    candidate_snapshot = rebuild_settings_snapshot(candidate)
    pending_rebuild = runtime_settings.pending_rebuild_setting_keys()
    changed = [
        key
        for key in sorted(normalized)
        if candidate_snapshot[key] != base_snapshot[key] or key in pending_rebuild
    ]
    if not changed:
        raise ValueError("Candidate settings are identical to the active rebuild slice")
    effective_overrides = {key: candidate_snapshot[key] for key in changed}
    return candidate, base_snapshot, candidate_snapshot, changed, bridge_identity


def _bridge_identity_hash(identity: dict[str, Any]) -> str:
    return _stable_hash(
        {
            "protocol_version": RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION,
            "identity": identity,
        }
    )


def _validated_closed_bridge_identity(identity: Any) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise RuntimeError("Candidate bridge identity must be an object")
    enabled = identity.get("bridge_enabled")
    common_keys = {
        "protocol_version",
        "bridge_enabled",
        "validation",
        "serving_identity_hash",
        "base_upstream_config_hash",
        "candidate_upstream_config_hash",
        "candidate_upstream_changed",
        "direct_graph_identity_hash",
        "embedding_api_protocol",
        "provider_response_used_as_fact",
        "model_call_count",
    }
    enabled_only = {
        "serving_chat_is_bridge_endpoint",
        "serving_embedding_is_bridge_endpoint",
    }
    expected_keys = common_keys | (enabled_only if enabled is True else set())
    if type(enabled) is not bool or set(identity) != expected_keys:
        raise RuntimeError("Candidate bridge identity violates the closed schema")
    if identity.get("protocol_version") != (
        RUNTIME_SETTINGS_BRIDGE_IDENTITY_PROTOCOL_VERSION
    ):
        raise RuntimeError("Candidate bridge identity protocol mismatch")
    if identity.get("provider_response_used_as_fact") is not False:
        raise RuntimeError("Provider response cannot be a bridge identity fact")
    if identity.get("model_call_count") != 0:
        raise RuntimeError("Bridge identity validation must have zero model calls")
    if identity.get("embedding_api_protocol") not in EMBEDDING_API_PROTOCOL_ALLOWLIST:
        raise RuntimeError("Candidate bridge embedding API protocol is invalid")
    _hash64(
        str(identity.get("serving_identity_hash") or ""),
        field_name="serving_identity_hash",
    )
    _hash64(
        str(identity.get("direct_graph_identity_hash") or ""),
        field_name="direct_graph_identity_hash",
    )
    if enabled:
        if identity.get("validation") != "upstream_provider_targets_validated":
            raise RuntimeError("Enabled bridge identity validation state is invalid")
        for key in (
            "base_upstream_config_hash",
            "candidate_upstream_config_hash",
        ):
            _hash64(str(identity.get(key) or ""), field_name=key)
        for key in enabled_only | {"candidate_upstream_changed"}:
            if type(identity.get(key)) is not bool:
                raise RuntimeError(f"Candidate bridge identity {key} must be boolean")
        if not all(identity[key] for key in enabled_only):
            raise RuntimeError(
                "Enabled bridge identity must bind both effective serving endpoints"
            )
        if identity["candidate_upstream_changed"] != (
            identity["candidate_upstream_config_hash"]
            != identity["base_upstream_config_hash"]
        ):
            raise RuntimeError(
                "Candidate bridge identity upstream-change fact is inconsistent"
            )
    else:
        if identity.get("validation") != "bypassed_bridge_disabled":
            raise RuntimeError("Disabled bridge identity validation state is invalid")
        if (
            identity.get("base_upstream_config_hash") is not None
            or identity.get("candidate_upstream_config_hash") is not None
            or identity.get("candidate_upstream_changed") is not False
        ):
            raise RuntimeError("Disabled bridge identity contains upstream facts")
    return dict(identity)


def _frozen_candidate_bridge_identity(
    candidate: RuntimeSettingsCandidate,
) -> dict[str, Any]:
    settings_json = dict(candidate.settings_json or {})
    diagnostics = dict(candidate.diagnostics_json or {})
    candidate_identity = diagnostics.get("candidate_identity")
    if not isinstance(candidate_identity, dict):
        raise RuntimeError("Runtime Settings candidate lost its immutable identity")
    cards = (
        settings_json.get("model_bridge_identity"),
        diagnostics.get("model_bridge_identity"),
        candidate_identity.get("model_bridge_identity"),
    )
    frozen = _validated_closed_bridge_identity(cards[0])
    if any(card != frozen for card in cards[1:]):
        raise RuntimeError("Runtime Settings candidate bridge identity copies drifted")
    expected_candidate_hash = _stable_hash(candidate_identity)
    if expected_candidate_hash != candidate.candidate_hash:
        raise RuntimeError("Runtime Settings candidate immutable hash drifted")
    if candidate_identity.get("candidate_overrides") != settings_json.get(
        "candidate_overrides"
    ):
        raise RuntimeError("Runtime Settings candidate overrides drifted")
    return frozen


def _bridge_identity_transition_matches(
    frozen: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    if frozen == current:
        return True
    if not frozen.get("bridge_enabled") or not current.get("bridge_enabled"):
        return False
    stable_keys = {
        "protocol_version",
        "bridge_enabled",
        "validation",
        "serving_identity_hash",
        "serving_chat_is_bridge_endpoint",
        "serving_embedding_is_bridge_endpoint",
        "direct_graph_identity_hash",
        "provider_response_used_as_fact",
        "model_call_count",
    }
    if any(frozen.get(key) != current.get(key) for key in stable_keys):
        return False
    expected_candidate_hash = frozen.get("candidate_upstream_config_hash")
    return bool(
        expected_candidate_hash
        and current.get("base_upstream_config_hash")
        == expected_candidate_hash
        and current.get("candidate_upstream_config_hash")
        == expected_candidate_hash
        and current.get("candidate_upstream_changed") is False
    )


def _assert_candidate_bridge_identity_current(
    candidate: RuntimeSettingsCandidate,
    *,
    stage: str,
    allow_activation_transition: bool = False,
) -> dict[str, Any]:
    try:
        frozen = _frozen_candidate_bridge_identity(candidate)
    except RuntimeError as exc:
        raise RuntimeSettingsBridgeIdentityDriftError(
            stage=stage,
            reason=f"frozen_candidate_identity_invalid:{type(exc).__name__}",
            expected_identity_hash=_stable_hash(
                {
                    "protocol_version": RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
                    "candidate_hash": str(candidate.candidate_hash or ""),
                }
            ),
            current_identity_hash=None,
        ) from None
    expected_hash = _bridge_identity_hash(frozen)
    overrides = dict((candidate.settings_json or {}).get("candidate_overrides") or {})
    try:
        current = _validated_candidate_bridge_identity(get_settings(), overrides)
        current = _validated_closed_bridge_identity(current)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeSettingsBridgeIdentityDriftError(
            stage=stage,
            reason=f"current_managed_identity_invalid:{type(exc).__name__}",
            expected_identity_hash=expected_hash,
            current_identity_hash=None,
        ) from None
    current_hash = _bridge_identity_hash(current)
    matches = frozen == current or (
        allow_activation_transition
        and _bridge_identity_transition_matches(frozen, current)
    )
    if not matches:
        raise RuntimeSettingsBridgeIdentityDriftError(
            stage=stage,
            reason="current_managed_identity_drifted",
            expected_identity_hash=expected_hash,
            current_identity_hash=current_hash,
        )
    return current


def _bridge_identity_failure_fact(
    error: RuntimeSettingsBridgeIdentityDriftError,
) -> dict[str, Any]:
    return {
        "protocol_version": BRIDGE_IDENTITY_FAILURE_PROTOCOL_VERSION,
        "stage": error.stage,
        "reason": error.reason,
        "expected_identity_hash": error.expected_identity_hash,
        "current_identity_hash": error.current_identity_hash,
        "model_call_count": 0,
        "provider_response_used_as_fact": False,
        "external_side_effects_started": False,
    }


def _record_bridge_identity_failure(
    candidate: RuntimeSettingsCandidate,
    *,
    error: RuntimeSettingsBridgeIdentityDriftError,
    builds: Sequence[RuntimeSettingsShadowBuild] = (),
    candidate_status: str | None = "promotion_blocked",
) -> dict[str, Any]:
    fact = _bridge_identity_failure_fact(error)
    if candidate_status is not None:
        candidate.status = candidate_status
    candidate.error_code = "bridge_identity_drift"
    candidate.last_error = error.reason[:128]
    candidate.blocking_reasons_json = sorted(
        set(candidate.blocking_reasons_json or []).union({"bridge_identity_drift"})
    )
    candidate.diagnostics_json = {
        **dict(candidate.diagnostics_json or {}),
        "bridge_identity_failure": fact,
    }
    for build in builds:
        if candidate_status is not None:
            build.status = (
                "failed" if candidate_status == "failed" else "promotion_blocked"
            )
        build.error_code = "bridge_identity_drift"
        build.last_error = error.reason[:128]
        build.blocking_reasons_json = sorted(
            set(build.blocking_reasons_json or []).union(
                {"bridge_identity_drift"}
            )
        )
        build.diagnostics_json = {
            **dict(build.diagnostics_json or {}),
            "bridge_identity_failure": fact,
        }
    return fact


def _candidate_chunk_schema_version(
    candidate_settings: Settings,
    changed_keys: Sequence[str],
) -> str:
    if not CHUNK_REBUILD_KEYS.intersection(changed_keys):
        return CHUNK_SCHEMA_VERSION
    digest = _stable_hash(
        {
            "protocol_version": "runtime_settings_chunk_schema_identity_v1",
            "base_chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "fixed_chunk_size_tokens": candidate_settings.fixed_chunk_size_tokens,
            "fixed_chunk_overlap_tokens": candidate_settings.fixed_chunk_overlap_tokens,
        }
    )
    return f"chunk_schema_v1_rs_{digest[:20]}"


def _candidate_embedding_text_version(
    candidate_settings: Settings,
    changed_keys: Sequence[str],
    chunk_schema_version: str,
) -> str:
    if not (VECTOR_REBUILD_KEYS.intersection(changed_keys) or CHUNK_REBUILD_KEYS.intersection(changed_keys)):
        return CURRENT_EMBEDDING_TEXT_VERSION
    digest = _stable_hash(
        {
            "protocol_version": "runtime_settings_vector_input_identity_v1",
            "embedding_model": candidate_settings.embedding_model,
            "embedding_api_protocol": candidate_settings.embedding_api_protocol,
            "embedding_dimensions": candidate_settings.embedding_dimensions,
            "embedding_base_url": candidate_settings.embedding_base_url,
            "embedding_resolve_ip": candidate_settings.embedding_resolve_ip,
            "chunk_schema_version": chunk_schema_version,
            "contextual_text_version": CURRENT_EMBEDDING_TEXT_VERSION,
        }
    )
    return f"contextual_text_v2_rs_{digest[:16]}"


def _graph_state_ids(pointer: KnowledgeBaseVectorRuntimeState) -> dict[str, str | None]:
    return {
        "context": pointer.active_context_graph_state_id,
        "relation": pointer.active_chunk_relation_graph_state_id,
        "mid": pointer.active_mid_concept_state_id,
        "coarse": pointer.active_coarse_concept_state_id,
    }


def _graph_bundle_hash(db: Session, graph_ids: dict[str, str | None]) -> str:
    model_by_key = {
        "context": ContextGraphState,
        "relation": ChunkRelationGraphState,
        "mid": MidConceptState,
        "coarse": CoarseConceptState,
    }
    cards: dict[str, Any] = {}
    for key, model in model_by_key.items():
        state_id = graph_ids.get(key)
        row = db.get(model, state_id) if state_id else None
        if row is None:
            raise RuntimeError(f"Active graph bundle is missing {key}:{state_id}")
        cards[key] = {
            "id": row.id,
            "state": row.state,
            "state_hash": (
                row.context_graph_hash
                if isinstance(row, ContextGraphState)
                else row.state_hash
            ),
        }
    return _stable_hash(
        {
            "protocol_version": "runtime_settings_base_graph_bundle_v1",
            "states": cards,
        }
    )


def _candidate_effective_runtime_hash(overrides: dict[str, Any]) -> str:
    with use_runtime_settings_override(overrides):
        return runtime_settings_state_hash()


def _scope_filter_hash(knowledge_base_id: str, chunk_ids: Sequence[str]) -> str:
    return _stable_hash(
        {
            "protocol_version": "qdrant_knowledge_base_chunk_scope_filter_v1",
            "knowledge_base_id": knowledge_base_id,
            "chunk_ids": list(chunk_ids),
        }
    )


def _base_facts(db: Session, knowledge_base_id: str) -> dict[str, Any]:
    active_graph_admission_gate(db, knowledge_base_id)
    chunks = list(db.scalars(active_chunks_query(knowledge_base_id)).all())
    if not chunks:
        raise ValueError(
            f"Knowledge base {knowledge_base_id} has no active chunks for shadow rebuild"
        )
    target = resolve_active_vector_runtime_target(db, knowledge_base_id, for_update=True)
    pointer = db.get(KnowledgeBaseVectorRuntimeState, target.runtime_state_id)
    if pointer is None:
        raise RuntimeError("Active vector runtime pointer disappeared")
    graph_ids = _graph_state_ids(pointer)
    return {
        "chunks": chunks,
        "chunk_ids": sorted(str(chunk.id) for chunk in chunks),
        "chunk_scope_hash": compute_chunk_scope_hash(chunks),
        "vector_target": target,
        "pointer": pointer,
        "vector_state_hash": target.runtime_state_hash,
        "graph_state_ids": graph_ids,
        "graph_bundle_hash": _graph_bundle_hash(db, graph_ids),
    }


def preview_runtime_settings_candidate(
    db: Session,
    *,
    knowledge_base_ids: Sequence[str],
    requested_settings: dict[str, Any],
) -> dict[str, Any]:
    target_ids = sorted({str(value).strip() for value in knowledge_base_ids if str(value).strip()})
    if not target_ids:
        raise ValueError("knowledge_base_ids must contain at least one id")
    if len(target_ids) > MAX_CANDIDATE_KNOWLEDGE_BASES:
        raise ValueError(
            f"A candidate may target at most {MAX_CANDIDATE_KNOWLEDGE_BASES} knowledge bases"
        )
    (
        candidate_settings,
        base_snapshot,
        candidate_snapshot,
        changed_keys,
        bridge_identity,
    ) = (
        _validated_candidate_settings(requested_settings)
    )
    overrides = {key: candidate_snapshot[key] for key in changed_keys}
    effective_runtime_hash = _candidate_effective_runtime_hash(overrides)
    chunk_schema_version = _candidate_chunk_schema_version(
        candidate_settings, changed_keys
    )
    embedding_text_version = _candidate_embedding_text_version(
        candidate_settings, changed_keys, chunk_schema_version
    )
    vector_required = bool(
        CHUNK_REBUILD_KEYS.intersection(changed_keys)
        or VECTOR_REBUILD_KEYS.intersection(changed_keys)
    )
    bases: dict[str, Any] = {}
    for knowledge_base_id in target_ids:
        knowledge_base = db.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise ValueError(f"Unknown knowledge base: {knowledge_base_id}")
        facts = _base_facts(db, knowledge_base_id)
        active_versions = list(
            db.scalars(
                select(DocumentVersion)
                .join(Document, Document.id == DocumentVersion.document_id)
                .where(
                    Document.knowledge_base_id == knowledge_base_id,
                    Document.is_active.is_(True),
                    DocumentVersion.is_active.is_(True),
                )
                .order_by(Document.id.asc())
                .limit(MAX_SHADOW_DOCUMENTS_PER_KB + 1)
            ).all()
        )
        if len(active_versions) > MAX_SHADOW_DOCUMENTS_PER_KB:
            raise RuntimeError("Candidate dry-run refused an unbounded document scope")
        missing_storage = sorted(
            str(version.storage_path)
            for version in active_versions
            if not Path(version.storage_path).is_file()
        )
        if missing_storage:
            raise RuntimeError(
                "Candidate dry-run found missing immutable document storage: "
                + ", ".join(missing_storage[:8])
            )
        bases[knowledge_base_id] = {
            "knowledge_base_name": knowledge_base.name,
            "base_chunk_version": int(knowledge_base.current_chunk_version or 0),
            "candidate_chunk_version": int(knowledge_base.current_chunk_version or 0)
            + (1 if CHUNK_REBUILD_KEYS.intersection(changed_keys) else 0),
            "base_chunk_count": len(facts["chunks"]),
            "base_document_count": len(active_versions),
            "base_chunk_scope_hash": facts["chunk_scope_hash"],
            "base_vector_state_hash": facts["vector_state_hash"],
            "base_graph_bundle_hash": facts["graph_bundle_hash"],
            "base_graph_state_ids": facts["graph_state_ids"],
        }
    base_runtime_hash = _stable_hash(
        {
            "protocol_version": "runtime_settings_rebuild_slice_v1",
            "settings": base_snapshot,
        }
    )
    preview = {
        "protocol_version": RUNTIME_SETTINGS_DRY_RUN_PROTOCOL_VERSION,
        "target_knowledge_base_ids": target_ids,
        "changed_keys": changed_keys,
        "base_runtime_version_hash": base_runtime_hash,
        "effective_runtime_settings_hash": effective_runtime_hash,
        "base_rebuild_settings": base_snapshot,
        "candidate_rebuild_settings": candidate_snapshot,
        "candidate_overrides": overrides,
        "model_bridge_identity": bridge_identity,
        "candidate_chunk_schema_version": chunk_schema_version,
        "candidate_embedding_text_version": embedding_text_version,
        "requires_shadow_rechunk": bool(CHUNK_REBUILD_KEYS.intersection(changed_keys)),
        "requires_vector_shadow": vector_required,
        "requires_four_layer_shadow": True,
        "fallback_enabled": False,
        "bounds": {
            "max_knowledge_bases": MAX_CANDIDATE_KNOWLEDGE_BASES,
            "max_documents_per_kb": MAX_SHADOW_DOCUMENTS_PER_KB,
            "max_chunks_per_kb": MAX_SHADOW_CHUNKS_PER_KB,
        },
        "knowledge_bases": bases,
        "active_env_mutated": False,
        "runtime_version_broadcast": False,
        "model_fallback_allowed": False,
        "database_fallback_allowed": False,
        "gray_zone_rule_decision_model_call_count": 0,
    }
    preview["dry_run_hash"] = _stable_hash(preview)
    return preview


def stage_runtime_settings_candidate(
    db: Session,
    *,
    knowledge_base_ids: Sequence[str],
    requested_settings: dict[str, Any],
    source: str = "api",
) -> tuple[RuntimeSettingsCandidate, list[RuntimeSettingsShadowBuild]]:
    preview = preview_runtime_settings_candidate(
        db,
        knowledge_base_ids=knowledge_base_ids,
        requested_settings=requested_settings,
    )
    frozen_stage_identity = _validated_closed_bridge_identity(
        preview["model_bridge_identity"]
    )
    expected_stage_identity_hash = _bridge_identity_hash(frozen_stage_identity)
    try:
        current_stage_identity = _validated_closed_bridge_identity(
            _validated_candidate_bridge_identity(
                get_settings(), dict(preview["candidate_overrides"])
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise RuntimeSettingsBridgeIdentityDriftError(
            stage="stage",
            reason=f"current_managed_identity_invalid:{type(exc).__name__}",
            expected_identity_hash=expected_stage_identity_hash,
            current_identity_hash=None,
        ) from None
    if current_stage_identity != frozen_stage_identity:
        raise RuntimeSettingsBridgeIdentityDriftError(
            stage="stage",
            reason="current_managed_identity_drifted",
            expected_identity_hash=expected_stage_identity_hash,
            current_identity_hash=_bridge_identity_hash(current_stage_identity),
        )
    target_ids = list(preview["target_knowledge_base_ids"])
    base_cards = dict(preview["knowledge_bases"])
    candidate_identity = {
        "protocol_version": RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
        "base_runtime_version_hash": preview["base_runtime_version_hash"],
        "effective_runtime_settings_hash": preview["effective_runtime_settings_hash"],
        "changed_keys": preview["changed_keys"],
        "candidate_overrides": preview["candidate_overrides"],
        "model_bridge_identity": preview["model_bridge_identity"],
        "target_knowledge_base_ids": target_ids,
        "base_cards": base_cards,
    }
    candidate_hash = _stable_hash(candidate_identity)
    existing = db.scalar(
        select(RuntimeSettingsCandidate).where(
            RuntimeSettingsCandidate.candidate_hash == candidate_hash
        )
    )
    if existing is not None:
        builds = list(
            db.scalars(
                select(RuntimeSettingsShadowBuild)
                .where(
                    RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                    == existing.id
                )
                .order_by(RuntimeSettingsShadowBuild.knowledge_base_id.asc())
            ).all()
        )
        if existing.status in {"staged", "building", "evaluating", "evaluation_passed", "promotion_blocked"}:
            return existing, builds
        raise RuntimeError(
            f"The exact Runtime Settings candidate is already terminal as {existing.status}"
        )

    live_general = list(
        db.scalars(
            select(RuntimeSettingsShadowBuild)
            .where(
                RuntimeSettingsShadowBuild.knowledge_base_id.in_(target_ids),
                RuntimeSettingsShadowBuild.status.in_(
                    {
                        "staged",
                        "dry_run_passed",
                        "building",
                        "shadow_ready",
                        "evaluating",
                        "evaluation_passed",
                        "promotion_blocked",
                        "promoting",
                    }
                ),
            )
            .with_for_update()
        ).all()
    )
    if live_general:
        raise RuntimeError(
            "A live Runtime Settings shadow build already owns a target knowledge base: "
            + ", ".join(
                f"{row.knowledge_base_id}:{row.id}:{row.status}"
                for row in live_general
            )
        )
    if preview["requires_vector_shadow"]:
        live_vectors = list(
            db.scalars(
                select(VectorShadowBuild)
                .where(
                    VectorShadowBuild.knowledge_base_id.in_(target_ids),
                    VectorShadowBuild.status.in_(LIVE_SHADOW_BUILD_STATUSES),
                )
                .with_for_update()
            ).all()
        )
        if live_vectors:
            raise RuntimeError(
                "A live vector shadow build already owns a target knowledge base: "
                + ", ".join(
                    f"{row.knowledge_base_id}:{row.id}:{row.status}"
                    for row in live_vectors
                )
            )

    candidate = RuntimeSettingsCandidate(
        protocol_version=RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
        candidate_hash=candidate_hash,
        base_runtime_version_hash=preview["base_runtime_version_hash"],
        settings_json={
            "protocol_version": RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
            "candidate_overrides": preview["candidate_overrides"],
            "model_bridge_identity": preview["model_bridge_identity"],
            "base_rebuild_settings": preview["base_rebuild_settings"],
            "candidate_rebuild_settings": preview["candidate_rebuild_settings"],
            "candidate_chunk_schema_version": preview[
                "candidate_chunk_schema_version"
            ],
            "candidate_embedding_text_version": preview[
                "candidate_embedding_text_version"
            ],
        },
        changed_keys_json=list(preview["changed_keys"]),
        target_knowledge_base_ids_json=target_ids,
        lifecycle_scope="rebuild_required",
        status="staged",
        source=str(source or "api")[:64],
        diagnostics_json={
            "protocol_version": RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
            "candidate_identity": candidate_identity,
            "effective_runtime_settings_hash": preview[
                "effective_runtime_settings_hash"
            ],
            "dry_run_hash": preview["dry_run_hash"],
            "model_bridge_identity": preview["model_bridge_identity"],
            "active_env_mutated": False,
            "runtime_version_broadcast": False,
            "gray_zone_rule_decision_model_call_count": 0,
        },
        blocking_reasons_json=[],
    )
    db.add(candidate)
    db.flush()

    candidate_settings = Settings.model_validate(
        {
            **get_settings().model_dump(mode="python"),
            **dict(preview["candidate_overrides"]),
        }
    )
    schema = frozen_vector_schema(
        embedding_api_protocol=candidate_settings.embedding_api_protocol,
        embedding_model=candidate_settings.embedding_model,
        embedding_dimension=candidate_settings.embedding_dimensions,
        embedding_text_version=preview["candidate_embedding_text_version"],
        chunk_schema_version=preview["candidate_chunk_schema_version"],
    )
    schema_json = schema.model_dump(mode="json")
    schema_hash = vector_schema_hash(schema)
    if preview["requires_vector_shadow"]:
        from app.services.vector_collection_cleanup import (
            assert_vector_collection_not_pending_cleanup,
        )

        assert_vector_collection_not_pending_cleanup(db, schema.collection_name)

    general_builds: list[RuntimeSettingsShadowBuild] = []
    for knowledge_base_id in target_ids:
        base = dict(base_cards[knowledge_base_id])
        vector_build: VectorShadowBuild | None = None
        if preview["requires_vector_shadow"]:
            vector_build = VectorShadowBuild(
                runtime_settings_candidate_id=candidate.id,
                knowledge_base_id=knowledge_base_id,
                protocol_version=VECTOR_SHADOW_BUILD_PROTOCOL_VERSION,
                status="staged",
                base_vector_state_hash=base["base_vector_state_hash"],
                candidate_vector_schema_json=schema_json,
                candidate_vector_schema_hash=schema_hash,
                embedding_model=schema.embedding_model,
                embedding_dimension=schema.embedding_dimension,
                distance_metric=schema.distance_metric,
                embedding_text_version=schema.embedding_text_version,
                chunk_schema_version=schema.chunk_schema_version,
                collection_identity_protocol_version=(
                    schema.collection_identity_protocol_version
                ),
                collection_identity_digest=schema.collection_identity_digest,
                collection_name=schema.collection_name,
                expected_point_count=int(base["base_chunk_count"]),
                ready_point_count=0,
                qdrant_proof_json={},
                evaluation_result_json={},
                promotion_audit_json={},
                rollback_audit_json={},
                diagnostics_json={
                    "staged_active_chunk_scope_hash_protocol_version": (
                        CHUNK_SCOPE_HASH_PROTOCOL_VERSION
                    ),
                    "staged_active_chunk_scope_hash": base[
                        "base_chunk_scope_hash"
                    ],
                    "base_active_expected_point_count": int(
                        base["base_chunk_count"]
                    ),
                    "staged_scope_filter_hash": _scope_filter_hash(
                        knowledge_base_id,
                        sorted(
                            str(chunk.id)
                            for chunk in db.scalars(
                                active_chunks_query(knowledge_base_id)
                            ).all()
                        ),
                    ),
                    "candidate_hash": candidate.candidate_hash,
                    "effective_runtime_settings_hash": preview[
                        "effective_runtime_settings_hash"
                    ],
                    "active_pointer_mutated": False,
                },
                blocking_reasons_json=[],
            )
            db.add(vector_build)
            db.flush()
        general_build = RuntimeSettingsShadowBuild(
            runtime_settings_candidate_id=candidate.id,
            knowledge_base_id=knowledge_base_id,
            vector_shadow_build_id=vector_build.id if vector_build else None,
            protocol_version=RUNTIME_SETTINGS_SHADOW_BUILD_PROTOCOL_VERSION,
            status="dry_run_passed",
            base_runtime_version_hash=preview["base_runtime_version_hash"],
            base_chunk_scope_hash=base["base_chunk_scope_hash"],
            base_vector_state_hash=base["base_vector_state_hash"],
            base_graph_bundle_hash=base["base_graph_bundle_hash"],
            base_graph_state_ids_json=dict(base["base_graph_state_ids"]),
            base_chunk_version=int(base["base_chunk_version"]),
            candidate_chunk_version=int(base["candidate_chunk_version"]),
            candidate_chunk_schema_version=preview[
                "candidate_chunk_schema_version"
            ],
            candidate_chunk_ids_json=[],
            candidate_document_version_ids_json=[],
            dry_run_json={
                **preview,
                "knowledge_bases": {knowledge_base_id: base},
            },
            dry_run_hash=preview["dry_run_hash"],
            build_metrics_json={},
            evaluation_result_json={},
            promotion_audit_json={},
            rollback_audit_json={},
            diagnostics_json={
                "candidate_hash": candidate.candidate_hash,
                "effective_runtime_settings_hash": preview[
                    "effective_runtime_settings_hash"
                ],
                "requires_shadow_rechunk": preview["requires_shadow_rechunk"],
                "requires_vector_shadow": preview["requires_vector_shadow"],
                "active_pointer_mutated": False,
                "active_env_mutated": False,
            },
            blocking_reasons_json=[],
        )
        db.add(general_build)
        general_builds.append(general_build)
    db.flush()
    return candidate, general_builds


def _locked_general_build(
    db: Session, build_id: str
) -> tuple[RuntimeSettingsShadowBuild, RuntimeSettingsCandidate]:
    build = db.scalar(
        select(RuntimeSettingsShadowBuild)
        .where(RuntimeSettingsShadowBuild.id == build_id)
        .with_for_update()
    )
    if build is None:
        raise ValueError(f"Unknown Runtime Settings shadow build: {build_id}")
    candidate = db.scalar(
        select(RuntimeSettingsCandidate)
        .where(RuntimeSettingsCandidate.id == build.runtime_settings_candidate_id)
        .with_for_update()
    )
    if candidate is None:
        raise RuntimeError("Runtime Settings shadow build lost its candidate")
    return build, candidate


def _assert_base_unchanged(
    db: Session,
    build: RuntimeSettingsShadowBuild,
    candidate: RuntimeSettingsCandidate,
    *,
    stage: str,
) -> dict[str, Any]:
    _assert_candidate_bridge_identity_current(candidate, stage=stage)
    active_rebuild_hash = _stable_hash(
        {
            "protocol_version": "runtime_settings_rebuild_slice_v1",
            "settings": rebuild_settings_snapshot(),
        }
    )
    if active_rebuild_hash != build.base_runtime_version_hash:
        raise RuntimeError("Active rebuild-required settings changed after candidate staging")
    facts = _base_facts(db, build.knowledge_base_id)
    mismatches = []
    if facts["chunk_scope_hash"] != build.base_chunk_scope_hash:
        mismatches.append("chunk_scope_hash")
    if facts["vector_state_hash"] != build.base_vector_state_hash:
        mismatches.append("vector_state_hash")
    if facts["graph_bundle_hash"] != build.base_graph_bundle_hash:
        mismatches.append("graph_bundle_hash")
    if facts["graph_state_ids"] != dict(build.base_graph_state_ids_json or {}):
        mismatches.append("graph_state_ids")
    if mismatches:
        raise RuntimeError(
            "Candidate base state changed after staging: " + ", ".join(mismatches)
        )
    expected_effective = str(
        (candidate.diagnostics_json or {}).get("effective_runtime_settings_hash")
        or ""
    )
    _hash64(expected_effective, field_name="effective_runtime_settings_hash")
    return facts


def _base_graph_reuse_states(
    db: Session,
    build: RuntimeSettingsShadowBuild,
) -> tuple[ChunkRelationGraphState, MidConceptState, CoarseConceptState]:
    graph_state_ids = dict(build.base_graph_state_ids_json or {})
    relation_state = db.get(
        ChunkRelationGraphState,
        str(graph_state_ids.get("relation") or ""),
    )
    mid_state = db.get(MidConceptState, str(graph_state_ids.get("mid") or ""))
    coarse_state = db.get(
        CoarseConceptState,
        str(graph_state_ids.get("coarse") or ""),
    )
    if relation_state is None or mid_state is None or coarse_state is None:
        raise RuntimeError(
            "Runtime Settings shadow build lost its active graph reuse source"
        )
    if (
        str(relation_state.knowledge_base_id) != str(build.knowledge_base_id)
        or str(mid_state.knowledge_base_id) != str(build.knowledge_base_id)
        or str(coarse_state.knowledge_base_id) != str(build.knowledge_base_id)
        or relation_state.state != "active"
        or mid_state.state != "active"
        or coarse_state.state != "active"
    ):
        raise RuntimeError(
            "Runtime Settings graph reuse source is not the active knowledge-base scope"
        )
    return relation_state, mid_state, coarse_state


def _concept_provider_build_evidence(
    db: Session,
    build: RuntimeSettingsShadowBuild,
    *,
    semantic_reuse_required: bool,
) -> dict[str, Any]:
    mid_state = db.get(MidConceptState, build.shadow_mid_concept_state_id)
    coarse_state = db.get(CoarseConceptState, build.shadow_coarse_concept_state_id)
    if mid_state is None or coarse_state is None:
        raise RuntimeError(
            "Runtime Settings shadow build lost its concept provider evidence"
        )

    def counts(state: MidConceptState | CoarseConceptState) -> dict[str, int]:
        stats = dict(state.stats_json or {})
        return {
            "semantic_reuse_hit_count": int(
                stats.get("provider_semantic_reuse_hit_count") or 0
            ),
            "semantic_reuse_miss_count": int(
                stats.get("provider_semantic_reuse_miss_count") or 0
            ),
            "provider_request_count": int(stats.get("provider_request_count") or 0),
        }

    mid = counts(mid_state)
    coarse = counts(coarse_state)
    evidence = {
        "protocol_version": "runtime_settings_concept_provider_evidence_v1",
        "semantic_reuse_required": bool(semantic_reuse_required),
        "mid": mid,
        "coarse": coarse,
        "semantic_reuse_hit_count": (
            mid["semantic_reuse_hit_count"]
            + coarse["semantic_reuse_hit_count"]
        ),
        "semantic_reuse_miss_count": (
            mid["semantic_reuse_miss_count"]
            + coarse["semantic_reuse_miss_count"]
        ),
        "provider_request_count": (
            mid["provider_request_count"] + coarse["provider_request_count"]
        ),
        "provider_response_persisted": False,
    }
    if semantic_reuse_required and (
        evidence["semantic_reuse_miss_count"] != 0
        or evidence["provider_request_count"] != 0
    ):
        raise RuntimeError(
            "Semantic-neutral Runtime Settings shadow build used the concept provider"
        )
    return {**evidence, "evidence_hash": _stable_hash(evidence)}


async def _build_concept_only_runtime_shadow(
    db: Session,
    *,
    build: RuntimeSettingsShadowBuild,
    chunks: list[Chunk],
    active_target: Any,
    base_relation_state: ChunkRelationGraphState,
    reuse_mid_state: MidConceptState,
    reuse_coarse_state: CoarseConceptState,
    shadow_metadata: dict[str, Any],
    require_provider_semantic_reuse: bool,
) -> ContextGraphState:
    base_context_state = db.get(
        ContextGraphState,
        str((build.base_graph_state_ids_json or {}).get("context") or ""),
    )
    if base_context_state is None or base_context_state.state != "active":
        raise RuntimeError(
            "Runtime Settings scoped shadow lost its active context proof source"
        )
    contextual_index_maintenance = json.loads(
        json.dumps(
            (base_context_state.diagnostics_json or {}).get(
                "contextual_index_maintenance"
            )
            or {},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )
    metadata = {
        **dict(shadow_metadata),
        **vector_runtime_diagnostics(active_target),
        "active_vector_reused_for_shadow": True,
        "active_vector_reuse_read_only": True,
        "runtime_settings_scoped_shadow_protocol_version": (
            "runtime_settings_concept_only_scoped_shadow_v1"
        ),
        "reused_active_graph_layers": ["relation"],
        "reused_active_relation_state_id": str(base_relation_state.id),
        "reused_active_relation_state_hash": str(base_relation_state.state_hash),
        "contextual_index_maintenance": contextual_index_maintenance,
    }
    profile = get_active_profile_record(db, build.knowledge_base_id)
    with use_strategy_profile(profile.profile_json):
        mid_state = await build_mid_concept_graph(
            db,
            build.knowledge_base_id,
            base_relation_state,
            state_scope="shadow",
            relation_state_scope="active",
            shadow_metadata=metadata,
            provider_semantic_reuse_source_state=reuse_mid_state,
            require_provider_semantic_reuse=require_provider_semantic_reuse,
        )
        coarse_state = await build_coarse_concept_graph(
            db,
            build.knowledge_base_id,
            mid_state,
            state_scope="shadow",
            relation_state_scope="active",
            shadow_metadata=metadata,
            provider_semantic_reuse_source_state=reuse_coarse_state,
            require_provider_semantic_reuse=require_provider_semantic_reuse,
        )
    context_state = write_context_graph_state(
        db,
        build.knowledge_base_id,
        base_relation_state,
        mid_state,
        coarse_state,
        chunks,
        state_scope="shadow",
        shadow_metadata=metadata,
        vector_runtime_target=active_target,
    )
    db.flush()
    return context_state


def _shadow_chunks(db: Session, build: RuntimeSettingsShadowBuild) -> list[Chunk]:
    ids = sorted(str(value) for value in (build.candidate_chunk_ids_json or []))
    if not ids:
        return []
    chunks = list(
        db.scalars(
            select(Chunk)
            .where(Chunk.id.in_(ids), Chunk.state == "shadow")
            .order_by(Chunk.id.asc())
        ).all()
    )
    if [str(chunk.id) for chunk in chunks] != ids:
        raise RuntimeError("Persisted shadow chunk scope is incomplete")
    if compute_chunk_scope_hash(chunks) != build.candidate_chunk_scope_hash:
        raise RuntimeError("Persisted shadow chunk scope hash drifted")
    return chunks


def _build_shadow_chunks(
    db: Session,
    *,
    build: RuntimeSettingsShadowBuild,
    candidate_settings: Settings,
) -> list[Chunk]:
    existing = _shadow_chunks(db, build)
    if existing:
        return existing
    document_rows = list(
        db.execute(
            select(Document, DocumentVersion)
            .join(
                DocumentVersion,
                DocumentVersion.document_id == Document.id,
            )
            .where(
                Document.knowledge_base_id == build.knowledge_base_id,
                Document.is_active.is_(True),
                DocumentVersion.is_active.is_(True),
            )
            .order_by(Document.id.asc())
            .limit(MAX_SHADOW_DOCUMENTS_PER_KB + 1)
        ).all()
    )
    if len(document_rows) > MAX_SHADOW_DOCUMENTS_PER_KB:
        raise RuntimeError("Shadow rechunk refused an unbounded document scope")
    knowledge_base = db.get(KnowledgeBase, build.knowledge_base_id)
    if knowledge_base is None:
        raise RuntimeError("Knowledge base disappeared during shadow rechunk")
    chunks: list[Chunk] = []
    candidate_version_ids: list[str] = []
    ingestion_root = get_settings().knowledge_base_paths_for_source_root(
        knowledge_base.source_root
    )["ingestion_root"]
    for document, active_version in document_rows:
        storage_path = Path(active_version.storage_path)
        frozen_snapshot = freeze_existing_source_snapshot(
            storage_path,
            authorized_root=ingestion_root,
            expected_checksum=active_version.checksum,
        )
        _source_type, sections = parse_document(frozen_snapshot)
        replay_frozen_source_snapshot(
            frozen_snapshot,
            authorized_root=ingestion_root,
        )
        candidate_version = DocumentVersion(
            document_id=document.id,
            version=build.candidate_chunk_version,
            checksum=active_version.checksum,
            storage_path=active_version.storage_path,
            extracted_path=active_version.extracted_path,
            parse_protocol_version=active_version.parse_protocol_version,
            language=active_version.language,
            language_source=active_version.language_source,
            language_confidence=active_version.language_confidence,
            language_detection_protocol_version=(
                active_version.language_detection_protocol_version
            ),
            language_detection_hash=active_version.language_detection_hash,
            language_metadata_json=dict(active_version.language_metadata_json or {}),
            is_active=False,
        )
        db.add(candidate_version)
        db.flush()
        candidate_version_ids.append(candidate_version.id)
        document_chunks = write_chunks_and_structure(
            db,
            knowledge_base=knowledge_base,
            document=document,
            version=candidate_version,
            sections=sections,
            chunk_version=build.candidate_chunk_version,
            chunk_size=candidate_settings.fixed_chunk_size_tokens,
            chunk_overlap=candidate_settings.fixed_chunk_overlap_tokens,
            chunk_state="shadow",
            deactivate_active=False,
            chunk_schema_version=build.candidate_chunk_schema_version,
            write_version_descriptor=False,
        )
        chunks.extend(document_chunks)
        if len(chunks) > MAX_SHADOW_CHUNKS_PER_KB:
            raise RuntimeError("Shadow rechunk exceeded its bounded chunk scope")
    if not chunks:
        raise RuntimeError("Shadow rechunk produced no chunks")
    chunks = sorted(chunks, key=lambda chunk: str(chunk.id))
    scope_hash = compute_chunk_scope_hash(chunks)
    descriptor = ChunkVersion(
        knowledge_base_id=build.knowledge_base_id,
        chunk_version=build.candidate_chunk_version,
        chunk_schema_version=build.candidate_chunk_schema_version,
        tokenizer_version=TOKENIZER_VERSION,
        chunk_size=candidate_settings.fixed_chunk_size_tokens,
        chunk_overlap=candidate_settings.fixed_chunk_overlap_tokens,
        state_hash=_stable_hash(
            {
                "protocol_version": "runtime_settings_shadow_chunk_version_v1",
                "candidate_id": build.runtime_settings_candidate_id,
                "knowledge_base_id": build.knowledge_base_id,
                "chunk_version": build.candidate_chunk_version,
                "chunk_schema_version": build.candidate_chunk_schema_version,
                "chunk_scope_hash": scope_hash,
            }
        ),
        stats_json={
            "chunk_count": len(chunks),
            "candidate_chunk_scope_hash": scope_hash,
        },
        diagnostics_json={
            "protocol_version": "runtime_settings_shadow_chunk_version_v1",
            "runtime_settings_candidate_id": build.runtime_settings_candidate_id,
            "state_scope": "shadow",
        },
        state="shadow",
    )
    db.add(descriptor)
    build.candidate_chunk_ids_json = [str(chunk.id) for chunk in chunks]
    build.candidate_document_version_ids_json = sorted(candidate_version_ids)
    build.candidate_chunk_scope_hash = scope_hash
    if build.vector_shadow_build_id:
        vector_build = db.get(VectorShadowBuild, build.vector_shadow_build_id)
        if vector_build is None:
            raise RuntimeError("Runtime Settings build lost its vector shadow build")
        vector_build.expected_point_count = len(chunks)
        vector_build.diagnostics_json = {
            **dict(vector_build.diagnostics_json or {}),
            "candidate_chunk_ids": [str(chunk.id) for chunk in chunks],
            "candidate_document_version_ids": sorted(candidate_version_ids),
            "candidate_chunk_scope_hash": scope_hash,
            "candidate_chunk_version": build.candidate_chunk_version,
            "base_chunk_version": build.base_chunk_version,
            "runtime_settings_shadow_build_id": build.id,
        }
    db.flush()
    return chunks


async def build_runtime_settings_shadow(
    db: Session,
    *,
    build_id: str,
    emit_heartbeats: bool = True,
) -> RuntimeSettingsShadowBuild:
    build, candidate = _locked_general_build(db, build_id)
    if build.status == "shadow_ready":
        _assert_base_unchanged(
            db, build, candidate, stage="build_idempotent"
        )
        return build
    if build.status not in {"dry_run_passed", "building", "failed"}:
        raise RuntimeError(
            f"Cannot build Runtime Settings shadow from status {build.status}"
        )
    _assert_base_unchanged(db, build, candidate, stage="build")
    overrides = dict((candidate.settings_json or {}).get("candidate_overrides") or {})
    if not overrides:
        raise RuntimeError("Runtime Settings candidate lost its overrides")
    build.status = "building"
    build.error_code = None
    build.last_error = None
    build.blocking_reasons_json = []
    build.started_at = build.started_at or datetime.utcnow()
    candidate.status = "building"
    candidate.error_code = None
    candidate.last_error = None
    candidate.blocking_reasons_json = []
    started = time.perf_counter()
    changed_keys = frozenset(str(key) for key in candidate.changed_keys_json or [])
    concept_only_shadow = bool(
        changed_keys
        and changed_keys.issubset(CONCEPT_SEMANTIC_NEUTRAL_REBUILD_KEYS)
        and not build.vector_shadow_build_id
        and not (build.diagnostics_json or {}).get("requires_shadow_rechunk")
    )
    semantic_reuse_required = bool(
        concept_only_shadow
    )
    relation_operating_point_evidence: dict[str, Any] = {
        "protocol_version": "runtime_settings_shadow_operating_point_reuse_v1",
        "active_operating_point_reused": False,
        "source_relation_state_id": None,
        "source_operating_point_hash": None,
        "changed_keys": sorted(changed_keys),
    }
    with use_runtime_settings_override(overrides) as candidate_settings:
        if (build.diagnostics_json or {}).get("requires_shadow_rechunk"):
            chunks = _build_shadow_chunks(
                db,
                build=build,
                candidate_settings=candidate_settings,
            )
        else:
            chunks = list(db.scalars(active_chunks_query(build.knowledge_base_id)).all())
        if build.vector_shadow_build_id:
            vector_build = await build_vector_shadow_artifacts(
                db,
                build_id=build.vector_shadow_build_id,
                emit_heartbeats=emit_heartbeats,
            )
            context_state = db.get(
                ContextGraphState, vector_build.shadow_context_graph_state_id
            )
            if context_state is None:
                raise RuntimeError("Vector shadow build did not produce a context graph")
            build.shadow_context_graph_state_id = vector_build.shadow_context_graph_state_id
            build.shadow_chunk_relation_graph_state_id = (
                vector_build.shadow_chunk_relation_graph_state_id
            )
            build.shadow_mid_concept_state_id = vector_build.shadow_mid_concept_state_id
            build.shadow_coarse_concept_state_id = (
                vector_build.shadow_coarse_concept_state_id
            )
            build.candidate_chunk_scope_hash = vector_build.chunk_scope_hash
        else:
            active_target = resolve_active_vector_runtime_target(
                db, build.knowledge_base_id, for_update=True
            )
            base_relation_state, reuse_mid_state, reuse_coarse_state = (
                _base_graph_reuse_states(db, build)
            )
            reuse_active_operating_point = not bool(
                changed_keys.intersection(RELATION_OPERATING_POINT_REBUILD_KEYS)
            )
            candidate_operating_point = (
                dict(base_relation_state.graph_operating_point_json or {})
                if reuse_active_operating_point
                else None
            )
            if reuse_active_operating_point:
                if not candidate_operating_point:
                    raise RuntimeError(
                        "Runtime Settings shadow build cannot reuse an empty active operating point"
                    )
                if graph_state_stable_hash(candidate_operating_point) != str(
                    base_relation_state.graph_operating_point_hash or ""
                ):
                    raise RuntimeError(
                        "Runtime Settings active operating point hash drifted"
                    )
                relation_operating_point_evidence = {
                    **relation_operating_point_evidence,
                    "active_operating_point_reused": True,
                    "source_relation_state_id": str(base_relation_state.id),
                    "source_operating_point_hash": str(
                        base_relation_state.graph_operating_point_hash
                    ),
                }
            context_shadow_metadata = {
                "runtime_settings_candidate_id": candidate.id,
                "runtime_settings_candidate_hash": (
                    candidate.diagnostics_json or {}
                ).get("effective_runtime_settings_hash"),
                "runtime_settings_candidate_identity_hash": candidate.candidate_hash,
                "runtime_settings_shadow_build_id": build.id,
                "active_pointer_mutated": False,
                "relation_operating_point_evidence": (
                    relation_operating_point_evidence
                ),
            }
            if concept_only_shadow:
                context_state = await _build_concept_only_runtime_shadow(
                    db,
                    build=build,
                    chunks=chunks,
                    active_target=active_target,
                    base_relation_state=base_relation_state,
                    reuse_mid_state=reuse_mid_state,
                    reuse_coarse_state=reuse_coarse_state,
                    shadow_metadata=context_shadow_metadata,
                    require_provider_semantic_reuse=semantic_reuse_required,
                )
            else:
                context_state = await rebuild_context_graph(
                    db,
                    build.knowledge_base_id,
                    state_scope="shadow",
                    operating_point=candidate_operating_point,
                    vector_runtime_target=active_target,
                    allow_active_vector_reuse_for_shadow=True,
                    chunks_override=chunks,
                    shadow_metadata=context_shadow_metadata,
                    emit_heartbeats=emit_heartbeats,
                    provider_semantic_reuse_source_mid_state=reuse_mid_state,
                    provider_semantic_reuse_source_coarse_state=reuse_coarse_state,
                    require_provider_semantic_reuse=semantic_reuse_required,
                )
            build.shadow_context_graph_state_id = context_state.id
            build.shadow_chunk_relation_graph_state_id = (
                context_state.chunk_relation_graph_state_id
            )
            build.shadow_mid_concept_state_id = context_state.mid_concept_state_id
            build.shadow_coarse_concept_state_id = context_state.coarse_concept_state_id
            build.candidate_chunk_scope_hash = context_state.chunk_scope_hash
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    concept_provider_evidence = _concept_provider_build_evidence(
        db,
        build,
        semantic_reuse_required=semantic_reuse_required,
    )
    build_metrics = {
        "protocol_version": "runtime_settings_shadow_build_metrics_v1",
        "elapsed_ms": round(elapsed_ms, 3),
        "chunk_count": len(chunks),
        "document_version_count": len(build.candidate_document_version_ids_json or []),
        "vector_shadow": bool(build.vector_shadow_build_id),
        "shadow_rechunk": bool(
            (build.diagnostics_json or {}).get("requires_shadow_rechunk")
        ),
        "context_graph_state_id": build.shadow_context_graph_state_id,
        "concept_provider_evidence": concept_provider_evidence,
        "relation_operating_point_evidence": relation_operating_point_evidence,
        "scoped_shadow_rebuild": bool(concept_only_shadow),
        "reused_active_graph_layers": ["relation"] if concept_only_shadow else [],
    }
    build.build_metrics_json = build_metrics
    build.build_result_hash = _stable_hash(
        {
            **build_metrics,
            "candidate_chunk_scope_hash": build.candidate_chunk_scope_hash,
            "graph_state_ids": {
                "context": build.shadow_context_graph_state_id,
                "relation": build.shadow_chunk_relation_graph_state_id,
                "mid": build.shadow_mid_concept_state_id,
                "coarse": build.shadow_coarse_concept_state_id,
            },
        }
    )
    build.status = "shadow_ready"
    build.shadow_ready_at = datetime.utcnow()
    candidate.status = "building"
    sibling_builds = list(
        db.scalars(
            select(RuntimeSettingsShadowBuild).where(
                RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                == candidate.id
            )
        ).all()
    )
    sibling_statuses = [row.status for row in sibling_builds]
    if "failed" in sibling_statuses:
        candidate.status = "failed"
        candidate.blocking_reasons_json = sorted(
            {
                reason
                for row in sibling_builds
                for reason in (row.blocking_reasons_json or [])
            }
        )
    elif sibling_statuses and all(
        status == "shadow_ready" for status in sibling_statuses
    ):
        candidate.status = "evaluating"
    db.flush()
    return build


def record_runtime_settings_build_failure(
    db: Session,
    *,
    build_id: str,
    error_type: str,
) -> RuntimeSettingsShadowBuild:
    build, candidate = _locked_general_build(db, build_id)
    if str(error_type or "") == "RuntimeSettingsBridgeIdentityDriftError":
        try:
            expected_hash = _bridge_identity_hash(
                _frozen_candidate_bridge_identity(candidate)
            )
        except RuntimeError:
            expected_hash = _stable_hash(
                {
                    "protocol_version": RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION,
                    "candidate_hash": str(candidate.candidate_hash or ""),
                }
            )
        error = RuntimeSettingsBridgeIdentityDriftError(
            stage="build",
            reason="current_managed_identity_rejected",
            expected_identity_hash=expected_hash,
            current_identity_hash=None,
        )
        _record_bridge_identity_failure(
            candidate,
            error=error,
            builds=[build],
            candidate_status="failed",
        )
        db.flush()
        return build
    build.status = "failed"
    build.error_code = "shadow_build_failed"
    build.last_error = str(error_type or "Exception")[:128]
    build.blocking_reasons_json = ["shadow_build_failed"]
    candidate.status = "failed"
    candidate.error_code = "shadow_build_failed"
    candidate.last_error = str(error_type or "Exception")[:128]
    candidate.blocking_reasons_json = ["shadow_build_failed"]
    db.flush()
    return build


def _evaluation_input_hash(
    build: RuntimeSettingsShadowBuild,
    context: ContextGraphState,
) -> str:
    return _stable_hash(
        {
            "protocol_version": RUNTIME_SETTINGS_EVALUATION_PROTOCOL_VERSION,
            "runtime_settings_candidate_id": build.runtime_settings_candidate_id,
            "build_id": build.id,
            "knowledge_base_id": build.knowledge_base_id,
            "dry_run_hash": build.dry_run_hash,
            "build_result_hash": build.build_result_hash,
            "candidate_chunk_scope_hash": build.candidate_chunk_scope_hash,
            "context_graph_state_id": context.id,
            "context_graph_hash": context.context_graph_hash,
        }
    )


def evaluate_runtime_settings_shadow(
    db: Session,
    *,
    build_id: str,
) -> RuntimeSettingsShadowBuild:
    build, candidate = _locked_general_build(db, build_id)
    if build.status == "evaluation_passed":
        try:
            _assert_base_unchanged(
                db, build, candidate, stage="evaluate_idempotent"
            )
        except RuntimeSettingsBridgeIdentityDriftError as exc:
            _record_bridge_identity_failure(
                candidate,
                error=exc,
                builds=[build],
            )
            db.flush()
        return build
    if build.status not in {"shadow_ready", "evaluating"}:
        raise RuntimeError(f"Cannot evaluate shadow build from {build.status}")
    try:
        _assert_base_unchanged(db, build, candidate, stage="evaluate")
    except RuntimeSettingsBridgeIdentityDriftError as exc:
        _record_bridge_identity_failure(
            candidate,
            error=exc,
            builds=[build],
        )
        db.flush()
        return build
    context = db.get(ContextGraphState, build.shadow_context_graph_state_id)
    relation = db.get(
        ChunkRelationGraphState, build.shadow_chunk_relation_graph_state_id
    )
    if context is None or relation is None:
        raise RuntimeError("Shadow build graph bundle is incomplete")
    reused_active_layers = set(
        str(value)
        for value in (
            (build.build_metrics_json or {}).get("reused_active_graph_layers")
            or []
        )
    )
    if reused_active_layers - {"relation"}:
        raise RuntimeError("Shadow build declared unsupported reused active layers")
    expected_relation_state = (
        "active" if "relation" in reused_active_layers else "shadow"
    )
    if context.state != "shadow" or relation.state != expected_relation_state:
        raise RuntimeError("Evaluation graph state scope does not match its shadow plan")
    if "relation" in reused_active_layers and str(relation.id) != str(
        (build.base_graph_state_ids_json or {}).get("relation") or ""
    ):
        raise RuntimeError("Shadow build reused an unbound active relation state")
    chunk_ids = sorted(relation.active_chunk_ids_json or [])
    if not chunk_ids or context.chunk_scope_hash != build.candidate_chunk_scope_hash:
        raise RuntimeError("Shadow graph chunk scope is incomplete")
    chunk_count = len(chunk_ids)
    mapped_chunk_count = int(
        db.scalar(
            select(func.count(func.distinct(ChunkStructureMapping.chunk_id))).where(
                ChunkStructureMapping.chunk_id.in_(chunk_ids)
            )
        )
        or 0
    )
    span_rows = list(
        db.execute(
            select(ChunkSpan, Chunk)
            .join(Chunk, Chunk.id == ChunkSpan.chunk_id)
            .where(ChunkSpan.chunk_id.in_(chunk_ids))
        ).all()
    )
    valid_span_chunk_ids = {
        str(chunk.id)
        for span, chunk in span_rows
        if int(span.char_start) == int(chunk.char_start)
        and int(span.char_end) == int(chunk.char_end)
        and int(span.token_start) == int(chunk.token_start)
        and int(span.token_end) == int(chunk.token_end)
        and int(span.char_end) > int(span.char_start)
        and int(span.token_end) > int(span.token_start)
        and str((span.metadata_json or {}).get("text_hash") or "") == chunk.text_hash
    }
    relation_edges = list(
        db.scalars(
            select(ChunkRelationEdge).where(
                ChunkRelationEdge.graph_state_id == relation.id
            )
        ).all()
    )
    connected_chunk_ids = {
        str(value)
        for edge in relation_edges
        for value in (edge.source_chunk_id, edge.target_chunk_id)
    }
    structure_recovery = mapped_chunk_count / max(chunk_count, 1)
    retrieval_quality = (
        1.0
        if chunk_count <= 1
        else len(connected_chunk_ids.intersection(chunk_ids)) / chunk_count
    )
    citation_quality = len(valid_span_chunk_ids) / max(chunk_count, 1)
    elapsed_ms = float((build.build_metrics_json or {}).get("elapsed_ms") or 0.0)
    context_stats = dict(context.stats_json or {})
    dimension = 0
    qdrant_schema_match = True
    vector_record_coverage = True
    if build.vector_shadow_build_id:
        vector_build = db.get(VectorShadowBuild, build.vector_shadow_build_id)
        if vector_build is None:
            raise RuntimeError("Evaluation lost its vector shadow build")
        dimension = int(vector_build.embedding_dimension)
        qdrant_schema_match = bool(
            (vector_build.qdrant_proof_json or {}).get("verified")
            and vector_build.qdrant_proof_hash
        )
        vector_record_coverage = bool(
            vector_build.ready_point_count == vector_build.expected_point_count
            and vector_build.expected_point_count == chunk_count
        )
    else:
        target = resolve_active_vector_runtime_target(db, build.knowledge_base_id)
        dimension = int(target.schema.embedding_dimension)
    estimated_vector_bytes = chunk_count * dimension * 4
    resource_budget_bytes = 2 * 1024 * 1024 * 1024
    candidate_settings = Settings.model_validate(
        {
            **get_settings().model_dump(mode="python"),
            **dict((candidate.settings_json or {}).get("candidate_overrides") or {}),
        }
    )
    isolated_ratio = 1.0 - retrieval_quality
    hard_gates = {
        "base_state_unchanged": True,
        "qdrant_schema_match": qdrant_schema_match,
        "vector_record_coverage": vector_record_coverage,
        "structure_recovery": structure_recovery
        >= candidate_settings.operating_point_hard_gate_min_structure_recovery_rate,
        "retrieval_quality": isolated_ratio
        <= candidate_settings.operating_point_hard_gate_max_isolated_ratio,
        "citation_quality": citation_quality >= 0.95,
        "latency_budget": elapsed_ms
        <= candidate_settings.operating_point_hard_gate_max_candidate_latency_p95_ms,
        "resource_budget": estimated_vector_bytes <= resource_budget_bytes,
    }
    metrics = {
        "chunk_count": chunk_count,
        "relation_edge_count": len(relation_edges),
        "structure_mapping_chunk_count": mapped_chunk_count,
        "valid_citation_span_chunk_count": len(valid_span_chunk_ids),
        "structure_recovery_rate": round(structure_recovery, 8),
        "retrieval_connected_chunk_rate": round(retrieval_quality, 8),
        "isolated_chunk_ratio": round(isolated_ratio, 8),
        "citation_span_valid_rate": round(citation_quality, 8),
        "build_elapsed_ms": round(elapsed_ms, 3),
        "estimated_vector_bytes": estimated_vector_bytes,
        "resource_budget_bytes": resource_budget_bytes,
        "mid_concept_count": int(context_stats.get("mid_concepts") or 0),
        "coarse_concept_count": int(context_stats.get("coarse_concepts") or 0),
        "gray_zone_rule_decision_model_call_count": 0,
    }
    evidence_payloads = {
        "retrieval_quality": {
            "connected_chunk_ids": sorted(connected_chunk_ids.intersection(chunk_ids)),
            "chunk_ids": chunk_ids,
            "relation_state_hash": relation.state_hash,
        },
        "citation_quality": {
            "valid_span_chunk_ids": sorted(valid_span_chunk_ids),
            "chunk_ids": chunk_ids,
        },
        "latency": {
            "elapsed_ms": elapsed_ms,
            "build_result_hash": build.build_result_hash,
        },
        "resource_usage": {
            "estimated_vector_bytes": estimated_vector_bytes,
            "chunk_count": chunk_count,
            "embedding_dimension": dimension,
        },
    }
    evidence_hashes = {
        key: _stable_hash(value) for key, value in evidence_payloads.items()
    }
    evaluation_input_hash = _evaluation_input_hash(build, context)
    failed = sorted(key for key, passed in hard_gates.items() if not passed)
    result = {
        "protocol_version": RUNTIME_SETTINGS_EVALUATION_PROTOCOL_VERSION,
        "evaluation_input_hash": evaluation_input_hash,
        "hard_gates": hard_gates,
        "metrics": metrics,
        "evidence_hashes": evidence_hashes,
        "evidence_payload_hash": _stable_hash(evidence_payloads),
        "passed": not failed,
        "blocking_reasons": failed,
        "model_fallback_used": False,
        "database_fallback_used": False,
        "gray_zone_rule_decision_model_call_count": 0,
    }
    build.evaluation_protocol_version = RUNTIME_SETTINGS_EVALUATION_PROTOCOL_VERSION
    build.evaluation_input_hash = evaluation_input_hash
    build.evaluation_result_json = result
    build.evaluation_result_hash = _stable_hash(result)
    build.evaluated_at = datetime.utcnow()
    build.blocking_reasons_json = failed
    if failed:
        build.status = "promotion_blocked"
        candidate.status = "promotion_blocked"
        candidate.blocking_reasons_json = sorted(
            set(candidate.blocking_reasons_json or []).union(failed)
        )
        db.flush()
        return build
    if build.vector_shadow_build_id:
        vector_build = db.get(VectorShadowBuild, build.vector_shadow_build_id)
        if vector_build is None:
            raise RuntimeError("Evaluation lost its vector build")
        vector_evaluation = VectorShadowEvaluation(
            protocol_version=VECTOR_SHADOW_EVALUATION_PROTOCOL_VERSION,
            evaluation_input_hash=vector_shadow_evaluation_input_hash(
                vector_build, context
            ),
            hard_gates={
                key: hard_gates[key]
                for key in sorted(REQUIRED_EVALUATION_GATES)
            },
            evidence_hashes={
                key: evidence_hashes[key]
                for key in sorted(REQUIRED_EVALUATION_EVIDENCE)
            },
            metrics=metrics,
            evaluator_diagnostics={
                "source_protocol": RUNTIME_SETTINGS_EVALUATION_PROTOCOL_VERSION,
                "runtime_settings_shadow_build_id": build.id,
                "measured_not_client_asserted": True,
            },
        )
        record_vector_shadow_evaluation(
            db,
            build_id=vector_build.id,
            evaluation=vector_evaluation,
        )
    build.status = "evaluation_passed"
    build.blocking_reasons_json = []
    sibling_builds = list(
        db.scalars(
            select(RuntimeSettingsShadowBuild).where(
                RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                == candidate.id,
                RuntimeSettingsShadowBuild.id != build.id,
            )
        ).all()
    )
    sibling_statuses = [row.status for row in sibling_builds]
    if any(status in {"promotion_blocked", "failed"} for status in sibling_statuses):
        candidate.status = "promotion_blocked"
        candidate.blocking_reasons_json = sorted(
            {
                reason
                for row in sibling_builds
                for reason in (row.blocking_reasons_json or [])
            }
        )
    elif all(status == "evaluation_passed" for status in sibling_statuses):
        candidate.status = "evaluation_passed"
        candidate.blocking_reasons_json = []
    else:
        candidate.status = "evaluating"
    db.flush()
    return build


def _graph_bundle(
    db: Session,
    *,
    graph_ids: dict[str, str | None],
    expected_state: str | Mapping[str, str],
) -> dict[str, Any]:
    model_by_key = {
        "context": ContextGraphState,
        "relation": ChunkRelationGraphState,
        "mid": MidConceptState,
        "coarse": CoarseConceptState,
    }
    rows: dict[str, Any] = {}
    for key, model in model_by_key.items():
        layer_expected_state = (
            str(expected_state[key])
            if isinstance(expected_state, Mapping)
            else str(expected_state)
        )
        state_id = graph_ids.get(key)
        row = db.scalar(
            select(model).where(model.id == state_id).with_for_update()
        ) if state_id else None
        if row is None or row.state != layer_expected_state:
            raise RuntimeError(
                f"Graph bundle {key}:{state_id} is not {layer_expected_state}"
            )
        rows[key] = row
    return rows


def _create_activation_intent(
    db: Session,
    *,
    candidate: RuntimeSettingsCandidate,
    direction: str,
) -> RuntimeSettingsActivationIntent:
    if direction not in {"promotion", "rollback"}:
        raise ValueError("Unsupported Runtime Settings activation direction")
    settings_json = dict(candidate.settings_json or {})
    if direction == "promotion":
        values = dict(settings_json.get("candidate_overrides") or {})
        expected_status = "promoted"
    else:
        base = dict(settings_json.get("base_rebuild_settings") or {})
        values = {
            key: base[key]
            for key in candidate.changed_keys_json or []
            if key in base
        }
        expected_status = "rolled_back"
    settings_hash = _stable_hash(values)
    existing = db.scalar(
        select(RuntimeSettingsActivationIntent).where(
            RuntimeSettingsActivationIntent.runtime_settings_candidate_id
            == candidate.id,
            RuntimeSettingsActivationIntent.direction == direction,
        )
    )
    if existing is not None:
        if existing.settings_hash != settings_hash:
            raise RuntimeError("Activation intent settings drifted")
        return existing
    intent = RuntimeSettingsActivationIntent(
        runtime_settings_candidate_id=candidate.id,
        protocol_version=RUNTIME_SETTINGS_ACTIVATION_PROTOCOL_VERSION,
        direction=direction,
        status="pending",
        settings_json=values,
        settings_hash=settings_hash,
        changed_keys_json=sorted(values),
        expected_candidate_status=expected_status,
        attempt_count=0,
        audit_json={
            "protocol_version": RUNTIME_SETTINGS_ACTIVATION_PROTOCOL_VERSION,
            "model_bridge_identity_hash": _bridge_identity_hash(
                _frozen_candidate_bridge_identity(candidate)
            ),
            "shared_env_mutated": False,
            "runtime_version_broadcast": False,
            "transaction_boundary": "after_postgresql_promotion_commit",
        },
    )
    db.add(intent)
    db.flush()
    return intent


def promote_runtime_settings_candidate(
    db: Session,
    candidate_id: str,
) -> dict[str, Any]:
    candidate = db.scalar(
        select(RuntimeSettingsCandidate)
        .where(RuntimeSettingsCandidate.id == candidate_id)
        .with_for_update()
    )
    if candidate is None:
        raise ValueError(f"Unknown Runtime Settings candidate: {candidate_id}")
    if candidate.protocol_version != RUNTIME_SETTINGS_CANDIDATE_PROTOCOL_VERSION:
        raise RuntimeError("Use the vector lifecycle for a legacy vector-only candidate")
    if candidate.status == "promoted":
        try:
            _assert_candidate_bridge_identity_current(
                candidate,
                stage="promote_idempotent",
                allow_activation_transition=True,
            )
        except RuntimeSettingsBridgeIdentityDriftError as exc:
            fact = _record_bridge_identity_failure(
                candidate,
                error=exc,
                candidate_status=None,
            )
            db.flush()
            return {
                "protocol_version": RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION,
                "candidate_id": candidate.id,
                "promoted": False,
                "blockers": ["bridge_identity_drift"],
                "bridge_identity_failure": fact,
                "idempotent_replay": True,
            }
        builds = list(
            db.scalars(
                select(RuntimeSettingsShadowBuild)
                .where(
                    RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                    == candidate.id
                )
                .order_by(RuntimeSettingsShadowBuild.knowledge_base_id.asc())
                .with_for_update()
            ).all()
        )
        if builds and not any(build.vector_shadow_build_id for build in builds):
            for build in builds:
                pointer = db.scalar(
                    select(KnowledgeBaseVectorRuntimeState)
                    .where(
                        KnowledgeBaseVectorRuntimeState.knowledge_base_id
                        == build.knowledge_base_id
                    )
                    .with_for_update()
                )
                if pointer is None or pointer.runtime_settings_candidate_id != candidate.id:
                    raise RuntimeError(
                        "Graph-only promotion replay pointer ownership mismatch"
                    )
                current_ids = _graph_state_ids(pointer)
                candidate_ids = {
                    "context": build.shadow_context_graph_state_id,
                    "relation": build.shadow_chunk_relation_graph_state_id,
                    "mid": build.shadow_mid_concept_state_id,
                    "coarse": build.shadow_coarse_concept_state_id,
                }
                if current_ids != candidate_ids:
                    raise RuntimeError(
                        "Graph-only promotion replay graph identity mismatch"
                    )
                audit = dict(build.promotion_audit_json or {})
                previous_ids = dict(audit.get("previous_graph_state_ids") or {})
                reused_active_layers = set(
                    str(value)
                    for value in (audit.get("reused_active_graph_layers") or [])
                )
                current_bundle = _graph_bundle(
                    db, graph_ids=current_ids, expected_state="active"
                )
                previous_bundle = _graph_bundle(
                    db,
                    graph_ids=previous_ids,
                    expected_state={
                        layer: (
                            "active" if layer in reused_active_layers else "inactive"
                        )
                        for layer in ("context", "relation", "mid", "coarse")
                    },
                )
                _set_graph_bundle_state(
                    db,
                    {
                        layer: row if layer not in reused_active_layers else None
                        for layer, row in previous_bundle.items()
                    },
                    "inactive",
                )
                _set_graph_bundle_state(
                    db,
                    {
                        layer: row if layer not in reused_active_layers else None
                        for layer, row in current_bundle.items()
                    },
                    "active",
                )
                build.promotion_audit_json = {
                    **audit,
                    "child_state_reconciled_on_idempotent_replay": True,
                }
        intent = _create_activation_intent(
            db, candidate=candidate, direction="promotion"
        )
        return {
            "protocol_version": RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION,
            "candidate_id": candidate.id,
            "promoted": True,
            "idempotent_replay": True,
            "activation_intent_id": intent.id,
        }
    builds = list(
        db.scalars(
            select(RuntimeSettingsShadowBuild)
            .where(
                RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                == candidate.id
            )
            .order_by(RuntimeSettingsShadowBuild.knowledge_base_id.asc())
            .with_for_update()
        ).all()
    )
    if not builds or any(build.status != "evaluation_passed" for build in builds):
        raise RuntimeError("Every candidate shadow build must pass measured evaluation")
    try:
        for build in builds:
            _assert_base_unchanged(db, build, candidate, stage="promote")
    except RuntimeSettingsBridgeIdentityDriftError as exc:
        fact = _record_bridge_identity_failure(
            candidate,
            error=exc,
            builds=builds,
        )
        db.flush()
        return {
            "protocol_version": RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION,
            "candidate_id": candidate.id,
            "promoted": False,
            "blockers": ["bridge_identity_drift"],
            "bridge_identity_failure": fact,
        }
    vector_builds = [build for build in builds if build.vector_shadow_build_id]
    if vector_builds and len(vector_builds) != len(builds):
        raise RuntimeError("A multi-KB candidate cannot mix vector and graph-only promotion")
    if vector_builds:
        result = promote_vector_shadow_candidate(db, candidate.id)
        if not result.get("promoted"):
            candidate.status = "promotion_blocked"
            candidate.blocking_reasons_json = list(result.get("blockers") or [])
            db.flush()
            return {
                "protocol_version": RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION,
                "candidate_id": candidate.id,
                "promoted": False,
                "blockers": candidate.blocking_reasons_json,
            }
        promoted_at = datetime.utcnow()
        for build in builds:
            vector_build = db.get(VectorShadowBuild, build.vector_shadow_build_id)
            if vector_build is None or vector_build.status != "promoted":
                raise RuntimeError("Vector promotion did not atomically promote every build")
            build.status = "promoted"
            build.promoted_at = promoted_at
            build.promotion_audit_json = {
                "protocol_version": RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION,
                "vector_shadow_build_id": vector_build.id,
                "vector_promotion_audit": dict(vector_build.promotion_audit_json or {}),
            }
    else:
        promoted_at = datetime.utcnow()
        candidate.status = "promoting"
        for build in builds:
            pointer = db.scalar(
                select(KnowledgeBaseVectorRuntimeState)
                .where(
                    KnowledgeBaseVectorRuntimeState.knowledge_base_id
                    == build.knowledge_base_id
                )
                .with_for_update()
            )
            if pointer is None or pointer.state_hash != build.base_vector_state_hash:
                raise RuntimeError("Active vector pointer changed before graph-only promotion")
            previous_ids = _graph_state_ids(pointer)
            candidate_ids = {
                "context": build.shadow_context_graph_state_id,
                "relation": build.shadow_chunk_relation_graph_state_id,
                "mid": build.shadow_mid_concept_state_id,
                "coarse": build.shadow_coarse_concept_state_id,
            }
            reused_active_layers = set(
                str(value)
                for value in (
                    (build.build_metrics_json or {}).get(
                        "reused_active_graph_layers"
                    )
                    or []
                )
            )
            if reused_active_layers - {"relation"}:
                raise RuntimeError(
                    "Runtime Settings promotion declared unsupported reused layers"
                )
            for layer in reused_active_layers:
                if str(previous_ids.get(layer) or "") != str(
                    candidate_ids.get(layer) or ""
                ):
                    raise RuntimeError(
                        f"Runtime Settings reused {layer} state identity drifted"
                    )
            previous_bundle = _graph_bundle(
                db, graph_ids=previous_ids, expected_state="active"
            )
            candidate_bundle = _graph_bundle(
                db,
                graph_ids=candidate_ids,
                expected_state={
                    layer: (
                        "active" if layer in reused_active_layers else "shadow"
                    )
                    for layer in ("context", "relation", "mid", "coarse")
                },
            )
            _set_graph_bundle_state(
                db,
                {
                    layer: row if layer not in reused_active_layers else None
                    for layer, row in previous_bundle.items()
                },
                "inactive",
            )
            _set_graph_bundle_state(
                db,
                {
                    layer: row if layer not in reused_active_layers else None
                    for layer, row in candidate_bundle.items()
                },
                "active",
            )
            previous_pointer = {
                "runtime_settings_candidate_id": pointer.runtime_settings_candidate_id,
                "activation_generation": pointer.activation_generation,
                "state_hash": pointer.state_hash,
                "graph_state_ids": previous_ids,
            }
            pointer.runtime_settings_candidate_id = candidate.id
            pointer.activation_generation = int(pointer.activation_generation) + 1
            pointer.active_context_graph_state_id = candidate_ids["context"]
            pointer.active_chunk_relation_graph_state_id = candidate_ids["relation"]
            pointer.active_mid_concept_state_id = candidate_ids["mid"]
            pointer.active_coarse_concept_state_id = candidate_ids["coarse"]
            schema = frozen_vector_schema(
                embedding_api_protocol=str(
                    (
                        (candidate.settings_json or {}).get(
                            "candidate_rebuild_settings"
                        )
                        or {}
                    ).get("embedding_api_protocol")
                    or ""
                ),
                embedding_model=pointer.embedding_model,
                embedding_dimension=pointer.embedding_dimension,
                embedding_text_version=pointer.embedding_text_version,
                chunk_schema_version=pointer.chunk_schema_version,
            )
            pointer.state_hash = vector_runtime_state_hash(
                knowledge_base_id=pointer.knowledge_base_id,
                runtime_settings_candidate_id=candidate.id,
                activation_generation=pointer.activation_generation,
                schema=schema,
                graph_state_ids=candidate_ids,
            )
            from app.services.vector_shadow_lifecycle import (
                _bind_context_qdrant_proof_to_active_pointer,
            )

            for layer, row in candidate_bundle.items():
                if layer not in reused_active_layers:
                    row.diagnostics_json = {
                        **dict(row.diagnostics_json or {}),
                        "active_vector_runtime_state_id": pointer.id,
                        "active_vector_runtime_state_hash": pointer.state_hash,
                        "active_vector_schema_hash": pointer.vector_schema_hash,
                        "promoted_from_runtime_settings_shadow": True,
                    }
            _bind_context_qdrant_proof_to_active_pointer(
                candidate_bundle["context"], pointer
            )
            audit = {
                "protocol_version": RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION,
                "candidate_id": candidate.id,
                "knowledge_base_id": build.knowledge_base_id,
                "previous_pointer": previous_pointer,
                "promoted_pointer_state_hash": pointer.state_hash,
                "previous_graph_state_ids": previous_ids,
                "promoted_graph_state_ids": candidate_ids,
                "reused_active_graph_layers": sorted(reused_active_layers),
                "active_env_mutated": False,
                "promoted_at": promoted_at.isoformat(),
            }
            pointer.previous_state_json = {
                **dict(pointer.previous_state_json or {}),
                "runtime_settings_graph_only": audit,
            }
            pointer.promotion_audit_json = {
                **dict(pointer.promotion_audit_json or {}),
                "runtime_settings_graph_only": audit,
            }
            build.status = "promoted"
            build.promoted_at = promoted_at
            build.promotion_audit_json = audit
        candidate.status = "promoted"
        candidate.promoted_at = promoted_at
    candidate.status = "promoted"
    candidate.promoted_at = candidate.promoted_at or datetime.utcnow()
    intent = _create_activation_intent(
        db, candidate=candidate, direction="promotion"
    )
    candidate.diagnostics_json = {
        **dict(candidate.diagnostics_json or {}),
        "activation_intent_id": intent.id,
        "activation_status": intent.status,
        "active_env_mutated": False,
        "runtime_version_broadcast": False,
    }
    db.flush()
    return {
        "protocol_version": RUNTIME_SETTINGS_PROMOTION_PROTOCOL_VERSION,
        "candidate_id": candidate.id,
        "candidate_hash": candidate.candidate_hash,
        "promoted": True,
        "idempotent_replay": False,
        "knowledge_base_ids": [build.knowledge_base_id for build in builds],
        "activation_intent_id": intent.id,
        "activation_pending_post_commit": True,
        "active_env_mutated": False,
    }


def apply_runtime_settings_activation_intent(intent_id: str) -> dict[str, Any]:
    """Apply one durable post-commit env/version intent and make retries safe."""

    from app.db import SessionLocal
    from app.services.runtime_settings import (
        _apply_runtime_env,
        _update_env_file,
        mark_runtime_settings_rebuild_applied,
        publish_runtime_settings_version,
    )

    with SessionLocal() as db:
        intent = db.scalar(
            select(RuntimeSettingsActivationIntent)
            .where(RuntimeSettingsActivationIntent.id == intent_id)
            .with_for_update()
        )
        if intent is None:
            raise ValueError(f"Unknown Runtime Settings activation intent: {intent_id}")
        candidate = db.get(RuntimeSettingsCandidate, intent.runtime_settings_candidate_id)
        if candidate is None or candidate.status != intent.expected_candidate_status:
            raise RuntimeError("Activation intent candidate status no longer matches")
        if intent.status == "applied":
            return {
                "intent_id": intent.id,
                "status": intent.status,
                "runtime_version_hash": intent.runtime_version_hash,
                "idempotent_replay": True,
            }
        if intent.status == "applying":
            lease_started = intent.updated_at or intent.created_at
            lease_age = (datetime.utcnow() - lease_started).total_seconds()
            if lease_age < ACTIVATION_APPLY_LEASE_SECONDS:
                raise RuntimeError(
                    "Runtime Settings activation intent is already applying; "
                    f"intent_id={intent.id}; retry after the lease expires"
                )
        direction = str(intent.direction)
        try:
            _assert_candidate_bridge_identity_current(
                candidate,
                stage=f"activation_{direction}",
                allow_activation_transition=True,
            )
        except RuntimeSettingsBridgeIdentityDriftError as exc:
            fact = _record_bridge_identity_failure(
                candidate,
                error=exc,
                candidate_status=None,
            )
            intent.status = "failed"
            intent.attempt_count = int(intent.attempt_count or 0) + 1
            intent.last_error_type = type(exc).__name__[:128]
            intent.audit_json = {
                **dict(intent.audit_json or {}),
                "bridge_identity_failure": fact,
                "last_attempt_failed": True,
                "last_error_type": type(exc).__name__,
                "shared_env_mutated": False,
                "runtime_version_broadcast": False,
            }
            db.commit()
            raise RuntimeError(
                "Runtime Settings activation bridge identity failed closed; "
                f"intent_id={intent_id}; blocker=bridge_identity_drift"
            ) from None
        intent.status = "applying"
        intent.attempt_count = int(intent.attempt_count or 0) + 1
        intent.last_error_type = None
        # SessionLocal expires ORM attributes on commit.  Copy the complete
        # post-commit side-effect payload while the locked row is still bound;
        # never dereference a detached/expired intent after leaving this
        # transaction.
        updates = dict(intent.settings_json or {})
        db.commit()
    try:
        _update_env_file(updates)
        _apply_runtime_env(updates)
        message = publish_runtime_settings_version(
            changed_keys=[key.upper() for key in sorted(updates)],
            source=f"runtime_settings_{direction}",
            idempotency_key=intent_id,
        )
    except Exception as exc:
        with SessionLocal() as db:
            failed = db.get(RuntimeSettingsActivationIntent, intent_id)
            if failed is not None:
                failed.status = "failed"
                failed.last_error_type = type(exc).__name__[:128]
                failed.audit_json = {
                    **dict(failed.audit_json or {}),
                    "last_attempt_failed": True,
                    "last_error_type": type(exc).__name__,
                }
                db.commit()
        raise RuntimeError(
            "Runtime Settings activation failed after PostgreSQL promotion; "
            f"intent_id={intent_id}; error_type={type(exc).__name__}; retry the intent"
        ) from None
    with SessionLocal() as db:
        applied = db.scalar(
            select(RuntimeSettingsActivationIntent)
            .where(RuntimeSettingsActivationIntent.id == intent_id)
            .with_for_update()
        )
        if applied is None:
            raise RuntimeError("Activation intent disappeared after shared-env update")
        if applied.status != "applying":
            raise RuntimeError(
                "Activation intent state changed during shared-env update"
            )
        broadcast_confirmed = bool(
            message.get("runtime_version_broadcast")
            and not message.get("broadcast_pending")
            and not message.get("local_refresh_pending")
        )
        applied.status = "applied" if broadcast_confirmed else "failed"
        applied.runtime_version_hash = message["version_hash"]
        applied.applied_at = datetime.utcnow() if broadcast_confirmed else None
        applied.last_error_type = (
            None if broadcast_confirmed else "RuntimeVersionDeliveryPending"
        )
        applied.audit_json = {
            **dict(applied.audit_json or {}),
            "shared_env_mutated": True,
            "runtime_version_broadcast": broadcast_confirmed,
            "runtime_version_broadcast_pending": bool(
                message.get("broadcast_pending")
            ),
            "runtime_local_refresh_pending": bool(
                message.get("local_refresh_pending")
            ),
            "runtime_version_hash": message["version_hash"],
            "applied_at": (
                applied.applied_at.isoformat()
                if applied.applied_at is not None
                else None
            ),
        }
        candidate = db.get(RuntimeSettingsCandidate, applied.runtime_settings_candidate_id)
        if candidate is not None:
            candidate.diagnostics_json = {
                **dict(candidate.diagnostics_json or {}),
                "activation_status": applied.status,
                "activated_runtime_version_hash": message["version_hash"],
                "active_env_mutated": True,
                "runtime_version_broadcast": broadcast_confirmed,
            }
        if broadcast_confirmed and direction == "promotion":
            mark_runtime_settings_rebuild_applied(
                db,
                changed_keys=list(updates),
                runtime_version_hash=message["version_hash"],
            )
        db.commit()
    return {
        "intent_id": intent_id,
        "status": "applied" if broadcast_confirmed else "failed",
        "runtime_version_hash": message["version_hash"],
        "runtime_version_broadcast": broadcast_confirmed,
        "runtime_version_broadcast_pending": bool(
            message.get("broadcast_pending")
        ),
        "runtime_local_refresh_pending": bool(
            message.get("local_refresh_pending")
        ),
        "idempotent_replay": False,
    }


def reconcile_runtime_settings_activation_intents(
    *,
    candidate_id: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    from app.db import SessionLocal

    with SessionLocal() as db:
        statement = select(RuntimeSettingsActivationIntent).where(
            RuntimeSettingsActivationIntent.status.in_(
                {"pending", "applying", "failed"}
            )
        )
        if candidate_id:
            statement = statement.where(
                RuntimeSettingsActivationIntent.runtime_settings_candidate_id
                == candidate_id
            )
        intents = list(
            db.scalars(
                statement.order_by(RuntimeSettingsActivationIntent.created_at.asc()).limit(
                    MAX_ACTIVATION_INTENTS + 1
                )
            ).all()
        )
    if len(intents) > MAX_ACTIVATION_INTENTS:
        raise RuntimeError("Activation reconciliation refused an unbounded intent scan")
    targets = [
        {
            "id": intent.id,
            "candidate_id": intent.runtime_settings_candidate_id,
            "direction": intent.direction,
            "status": intent.status,
            "attempt_count": intent.attempt_count,
        }
        for intent in intents
    ]
    results: list[dict[str, Any]] = []
    if not dry_run:
        for intent in intents:
            results.append(apply_runtime_settings_activation_intent(intent.id))
    return {
        "protocol_version": "runtime_settings_activation_reconcile_v1",
        "dry_run": dry_run,
        "target_count": len(targets),
        "targets": targets,
        "results": results,
    }


def rollback_runtime_settings_candidate(
    db: Session,
    candidate_id: str,
    *,
    reason: str,
) -> RuntimeSettingsCandidate:
    if not str(reason or "").strip():
        raise ValueError("rollback reason is required")
    candidate = db.scalar(
        select(RuntimeSettingsCandidate)
        .where(RuntimeSettingsCandidate.id == candidate_id)
        .with_for_update()
    )
    if candidate is None:
        raise ValueError(f"Unknown Runtime Settings candidate: {candidate_id}")
    if candidate.status == "rolled_back":
        return candidate
    if candidate.status != "promoted":
        abandonable_statuses = {
            "staged",
            "building",
            "evaluating",
            "evaluation_passed",
            "promotion_blocked",
            "failed",
        }
        if candidate.status not in abandonable_statuses:
            raise RuntimeError(
                "Runtime Settings candidate is neither promoted nor safely abandonable"
            )
        builds = list(
            db.scalars(
                select(RuntimeSettingsShadowBuild)
                .where(
                    RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                    == candidate.id
                )
                .order_by(RuntimeSettingsShadowBuild.knowledge_base_id.asc())
                .with_for_update()
            ).all()
        )
        if any(build.vector_shadow_build_id for build in builds):
            raise RuntimeError(
                "Unpromoted vector candidates require the vector abandonment lifecycle"
            )
        model_by_layer = {
            "context": ContextGraphState,
            "relation": ChunkRelationGraphState,
            "mid": MidConceptState,
            "coarse": CoarseConceptState,
        }
        abandoned_from_status = str(candidate.status)
        for build in builds:
            _assert_base_unchanged(db, build, candidate, stage="abandon")
            base_ids = dict(build.base_graph_state_ids_json or {})
            candidate_ids = {
                "context": build.shadow_context_graph_state_id,
                "relation": build.shadow_chunk_relation_graph_state_id,
                "mid": build.shadow_mid_concept_state_id,
                "coarse": build.shadow_coarse_concept_state_id,
            }
            retired_layers: list[str] = []
            retired_bundle: dict[str, Any | None] = {
                layer: None for layer in model_by_layer
            }
            for layer, model in model_by_layer.items():
                state_id = candidate_ids.get(layer)
                if not state_id or str(state_id) == str(base_ids.get(layer) or ""):
                    continue
                row = db.scalar(
                    select(model).where(model.id == state_id).with_for_update()
                )
                if row is None:
                    raise RuntimeError(
                        f"Unpromoted Runtime Settings candidate lost {layer} shadow state"
                    )
                if row.state not in {"shadow", "inactive"}:
                    raise RuntimeError(
                        f"Unpromoted Runtime Settings {layer} state is unexpectedly {row.state}"
                    )
                retired_bundle[layer] = row
                retired_layers.append(layer)
            _set_graph_bundle_state(db, retired_bundle, "inactive")
            build.status = "rolled_back"
            build.rolled_back_at = datetime.utcnow()
            build.rollback_audit_json = {
                "protocol_version": RUNTIME_SETTINGS_ROLLBACK_PROTOCOL_VERSION,
                "reason": str(reason)[:500],
                "unpromoted_abandon": True,
                "abandoned_from_status": abandoned_from_status,
                "retired_shadow_layers": retired_layers,
                "active_pointer_mutated": False,
                "active_env_mutated": False,
            }
        candidate.status = "rolled_back"
        candidate.rolled_back_at = datetime.utcnow()
        candidate.diagnostics_json = {
            **dict(candidate.diagnostics_json or {}),
            "rollback_reason": str(reason)[:500],
            "unpromoted_abandon": True,
            "abandoned_from_status": abandoned_from_status,
            "active_pointer_mutated": False,
            "active_env_mutated": False,
            "runtime_version_broadcast": False,
        }
        db.flush()
        return candidate
    builds = list(
        db.scalars(
            select(RuntimeSettingsShadowBuild)
            .where(
                RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                == candidate.id
            )
            .order_by(RuntimeSettingsShadowBuild.knowledge_base_id.asc())
            .with_for_update()
        ).all()
    )
    if any(build.vector_shadow_build_id for build in builds):
        rollback_vector_shadow_candidate(db, candidate.id, reason=reason)
        for build in builds:
            build.status = "rolled_back"
            build.rolled_back_at = datetime.utcnow()
            build.rollback_audit_json = {
                "protocol_version": RUNTIME_SETTINGS_ROLLBACK_PROTOCOL_VERSION,
                "vector_shadow_build_id": build.vector_shadow_build_id,
                "reason": str(reason)[:500],
            }
    else:
        for build in builds:
            pointer = db.scalar(
                select(KnowledgeBaseVectorRuntimeState)
                .where(
                    KnowledgeBaseVectorRuntimeState.knowledge_base_id
                    == build.knowledge_base_id
                )
                .with_for_update()
            )
            audit = dict(build.promotion_audit_json or {})
            previous = dict(audit.get("previous_pointer") or {})
            if pointer is None or pointer.runtime_settings_candidate_id != candidate.id:
                raise RuntimeError("Graph-only rollback pointer ownership mismatch")
            current_ids = _graph_state_ids(pointer)
            previous_ids = dict(previous.get("graph_state_ids") or {})
            reused_active_layers = set(
                str(value)
                for value in (audit.get("reused_active_graph_layers") or [])
            )
            if reused_active_layers - {"relation"}:
                raise RuntimeError(
                    "Runtime Settings rollback declared unsupported reused layers"
                )
            for layer in reused_active_layers:
                if str(current_ids.get(layer) or "") != str(
                    previous_ids.get(layer) or ""
                ):
                    raise RuntimeError(
                        f"Runtime Settings rollback reused {layer} identity drifted"
                    )
            current_bundle = _graph_bundle(
                db, graph_ids=current_ids, expected_state="active"
            )
            previous_bundle = _graph_bundle(
                db,
                graph_ids=previous_ids,
                expected_state={
                    layer: (
                        "active" if layer in reused_active_layers else "inactive"
                    )
                    for layer in ("context", "relation", "mid", "coarse")
                },
            )
            _set_graph_bundle_state(
                db,
                {
                    layer: row if layer not in reused_active_layers else None
                    for layer, row in current_bundle.items()
                },
                "inactive",
            )
            _set_graph_bundle_state(
                db,
                {
                    layer: row if layer not in reused_active_layers else None
                    for layer, row in previous_bundle.items()
                },
                "active",
            )
            pointer.runtime_settings_candidate_id = previous.get(
                "runtime_settings_candidate_id"
            )
            pointer.activation_generation = int(previous["activation_generation"])
            pointer.active_context_graph_state_id = previous_ids["context"]
            pointer.active_chunk_relation_graph_state_id = previous_ids["relation"]
            pointer.active_mid_concept_state_id = previous_ids["mid"]
            pointer.active_coarse_concept_state_id = previous_ids["coarse"]
            pointer.state_hash = str(previous["state_hash"])
            from app.services.vector_shadow_lifecycle import (
                _bind_context_qdrant_proof_to_active_pointer,
            )

            _bind_context_qdrant_proof_to_active_pointer(
                previous_bundle["context"], pointer
            )
            build.status = "rolled_back"
            build.rolled_back_at = datetime.utcnow()
            build.rollback_audit_json = {
                "protocol_version": RUNTIME_SETTINGS_ROLLBACK_PROTOCOL_VERSION,
                "reason": str(reason)[:500],
                "restored_pointer_state_hash": pointer.state_hash,
                "restored_graph_state_ids": previous_ids,
                "reused_active_graph_layers": sorted(reused_active_layers),
            }
        candidate.status = "rolled_back"
        candidate.rolled_back_at = datetime.utcnow()
    candidate.status = "rolled_back"
    candidate.rolled_back_at = candidate.rolled_back_at or datetime.utcnow()
    intent = _create_activation_intent(db, candidate=candidate, direction="rollback")
    candidate.diagnostics_json = {
        **dict(candidate.diagnostics_json or {}),
        "rollback_reason": str(reason)[:500],
        "rollback_activation_intent_id": intent.id,
        "activation_status": intent.status,
    }
    db.flush()
    return candidate


def runtime_settings_candidate_payload(
    db: Session,
    candidate_id: str,
) -> dict[str, Any]:
    candidate = db.get(RuntimeSettingsCandidate, candidate_id)
    if candidate is None:
        raise ValueError(f"Unknown Runtime Settings candidate: {candidate_id}")
    builds = list(
        db.scalars(
            select(RuntimeSettingsShadowBuild)
            .where(
                RuntimeSettingsShadowBuild.runtime_settings_candidate_id
                == candidate.id
            )
            .order_by(RuntimeSettingsShadowBuild.knowledge_base_id.asc())
            .limit(MAX_STATUS_BUILDS + 1)
        ).all()
    )
    if len(builds) > MAX_STATUS_BUILDS:
        raise RuntimeError("Candidate status refused an unbounded build scan")
    intents = list(
        db.scalars(
            select(RuntimeSettingsActivationIntent)
            .where(
                RuntimeSettingsActivationIntent.runtime_settings_candidate_id
                == candidate.id
            )
            .order_by(RuntimeSettingsActivationIntent.created_at.asc())
        ).all()
    )
    return {
        "id": candidate.id,
        "protocol_version": candidate.protocol_version,
        "candidate_hash": candidate.candidate_hash,
        "effective_runtime_settings_hash": (
            candidate.diagnostics_json or {}
        ).get("effective_runtime_settings_hash"),
        "base_runtime_version_hash": candidate.base_runtime_version_hash,
        "status": candidate.status,
        "changed_keys": list(candidate.changed_keys_json or []),
        "target_knowledge_base_ids": list(
            candidate.target_knowledge_base_ids_json or []
        ),
        "settings": dict(candidate.settings_json or {}),
        "blocking_reasons": list(candidate.blocking_reasons_json or []),
        "diagnostics": dict(candidate.diagnostics_json or {}),
        "builds": [
            {
                "id": build.id,
                "knowledge_base_id": build.knowledge_base_id,
                "status": build.status,
                "vector_shadow_build_id": build.vector_shadow_build_id,
                "base_chunk_scope_hash": build.base_chunk_scope_hash,
                "candidate_chunk_scope_hash": build.candidate_chunk_scope_hash,
                "candidate_chunk_version": build.candidate_chunk_version,
                "candidate_chunk_schema_version": (
                    build.candidate_chunk_schema_version
                ),
                "shadow_context_graph_state_id": (
                    build.shadow_context_graph_state_id
                ),
                "build_metrics": dict(build.build_metrics_json or {}),
                "evaluation": dict(build.evaluation_result_json or {}),
                "blocking_reasons": list(build.blocking_reasons_json or []),
            }
            for build in builds
        ],
        "activation_intents": [
            {
                "id": intent.id,
                "direction": intent.direction,
                "status": intent.status,
                "attempt_count": intent.attempt_count,
                "runtime_version_hash": intent.runtime_version_hash,
                "last_error_type": intent.last_error_type,
            }
            for intent in intents
        ],
    }

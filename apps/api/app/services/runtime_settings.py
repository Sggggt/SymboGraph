from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import secrets
import stat
import tempfile
import time
import unicodedata
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse
from uuid import UUID

import httpx
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import (
    ACTIVE_RQ_KMEANS_LEVELS,
    EDGE_DISTANCE_PROTOCOL_ALLOWLIST,
    EDGE_PROJECTION_PROTOCOL_ALLOWLIST,
    EDGE_TYPE_CALIBRATION_PROTOCOL_ALLOWLIST,
    EMBEDDING_API_PROTOCOL_ALLOWLIST,
    GRAY_ZONE_OBSERVATION_CADENCE_MAX,
    GRAY_ZONE_RULE_PROTOCOL_ALLOWLIST,
    HOT_RELOAD_SETTINGS,
    MODEL_API_PROTOCOL_ALLOWLIST,
    PROCESS_ONLY_ENV_KEYS,
    REBUILD_REQUIRED_SETTINGS,
    RQ_MEMBERSHIP_PROTOCOL_ALLOWLIST,
    RUNTIME_ENV_SETTINGS,
    SERVICE_RECREATE_REQUIRED_SETTINGS,
    Settings,
    TRAVERSAL_OBSERVATION_BUDGET_MAX,
    WORKSPACE_ROOT,
    get_settings,
    model_bridge_client_base_url,
    runtime_settings_override_active,
    validate_path_distance_thresholds,
)
from app.services.error_sanitizer import public_exception_message, sanitize_error_message


ENV_PATH = Path(
    os.environ.get("RUNTIME_ENV_FILE") or (WORKSPACE_ROOT / ".env")
)
ENV_EXAMPLE_PATH = Path(ENV_PATH).with_name(".env.example")
RUNTIME_SETTINGS_AUDIT_PROTOCOL_VERSION = "runtime_settings_audit_v1"
RUNTIME_ENV_FILE_LOCK_PROTOCOL_VERSION = "runtime_env_file_lock_v1"
RUNTIME_ENV_FILE_IDENTITY_PROTOCOL_VERSION = "runtime_env_file_identity_v1"
RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION = "runtime_env_file_cas_v1"
RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION = "runtime_env_file_recovery_v1"
RUNTIME_ENV_FILE_LOCK_TIMEOUT_SECONDS = 30.0
RUNTIME_SETTINGS_PUBLICATION_ADVISORY_SCOPE = (
    "runtime_settings_version_publication_v1"
)
RUNTIME_ENV_FILE_LOCK_POLL_SECONDS = 0.05
SETTINGS_VERSION_REDIS_KEY = "symbograph:runtime_settings:version"
SETTINGS_VERSION_CHANNEL = "symbograph:runtime_settings:updated"
KNOWN_UNSAFE_BRIDGE_ADMIN_TOKENS = frozenset(
    {
        "change-me",
        "changeme",
        "default",
        "local-model-bridge-admin",
        "model-bridge-admin-token",
    }
)
DEPRECATED_ENV_KEYS: set[str] = {
    "CHUNK_TOKEN_BUDGET",
    "SEMANTIC_CHUNKING_ENABLED",
    "SEMANTIC_CHUNKING_MIN_LENGTH",
    "ENABLE_GRAPH_COMMUNITY_SUMMARIES",
    "COMMUNITY_LOUVAIN_RESOLUTION",
    "COMMUNITY_MIN_MODULARITY_WARN",
    "SIGNAL_EXTRACTION_MAX_MODEL_BATCHES",
    "SIGNAL_EXTRACTION_MAX_CANDIDATES_PER_BATCH",
    "SIGNAL_EXTRACTION_MAX_TOKENS_PER_BATCH",
    "SIGNAL_CANDIDATE_KEEP_THRESHOLD",
    "INGESTION_FILE_CONCURRENCY",
    "RETRIEVAL_RECALL_K_DEFAULT",
    "RETRIEVAL_RECALL_K_FORMULA",
    "RETRIEVAL_LAYER_ENABLED",
    "RETRIEVAL_CACHE_TTL_SECONDS",
    "ENABLE_AGENTIC_REFLECTION",
    "ENABLE_POST_GENERATION_REFLECTION",
    "CITATION_VERIFICATION_SAMPLE_MAX",
    "REFLECTION_MAX_RETRIES",
    "COARSE_COMMUNITY_RESOLUTION",
    "COARSE_COMMUNITY_MIN_MODULARITY_WARN",
    "GRAPH_OVERVIEW_MAX_NODES",
    "GRAPH_OVERVIEW_MAX_EDGES",
    "AGENT_COARSE_ACTIVATION_BUDGET",
    "AGENT_MID_ACTIVATION_BUDGET",
    "AGENT_FINE_CLUSTER_BUDGET",
    "AGENT_FINE_ENTRY_BUDGET",
    "AGENT_AMBIGUOUS_EDGE_DISTANCE_LOW",
    "AGENT_AMBIGUOUS_EDGE_DISTANCE_HIGH",
    "AGENT_COARSE_ENTRY_BUDGET",
    "AGENT_COARSE_JUMP_BUDGET",
    "AGENT_MID_ENTRY_BUDGET",
    "AGENT_MID_EXPANSION_RADIUS_CAP",
    "AGENT_FINE_ADDRESS_SEED_BUDGET",
    "AGENT_RQ_MEMBERSHIP_SEED_BUDGET",
    "AGENT_FRONTIER_EXPANSION_BUDGET",
    "AGENT_DRILLDOWN_BUDGET_PER_LAYER",
    "AGENT_CHUNK_CANDIDATE_BUDGET",
    "RERANKER_ENABLED",
    "RERANKER_MODEL",
    "RERANKER_MAX_LENGTH",
    "RERANKER_DEVICE",
    "HF_HUB_OFFLINE",
    "OPENAI_API_KEY",
}
_LAST_RUNTIME_SETTINGS_VERSION: str | None = None
_RUNTIME_ENV_PROCESS_APPLIED_VALUES: dict[str, str | None] = {}
SECRET_RUNTIME_SETTING_KEYS = frozenset(
    {"chat_api_key", "graph_api_key", "embedding_api_key"}
)
SECRET_CLEAR_CONTROL_KEYS = {
    "clear_chat_api_key": "chat_api_key",
    "clear_graph_api_key": "graph_api_key",
    "clear_embedding_api_key": "embedding_api_key",
}


def read_env_int(key: str, default: int = 0) -> int:
    """直接从 .env 文件读取整数值（热加载，绕过 os.environ 缓存）。"""
    if runtime_settings_override_active():
        return default
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def read_env_float(key: str, default: float = 0.0) -> float:
    if runtime_settings_override_active():
        return default
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_env_bool(key: str, default: bool = False) -> bool:
    if runtime_settings_override_active():
        return default
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def runtime_lifecycle_payload() -> dict:
    rebuild_required = sorted(REBUILD_REQUIRED_SETTINGS)
    hot_reloadable = sorted(HOT_RELOAD_SETTINGS)
    service_recreate_required = sorted(SERVICE_RECREATE_REQUIRED_SETTINGS)
    return {
        "hot_reloadable": hot_reloadable,
        "rebuild_required": rebuild_required,
        "service_recreate_required": service_recreate_required,
        "candidate_version_required_for": rebuild_required,
        "fixed_protocol": {"rq_kmeans_levels": ACTIVE_RQ_KMEANS_LEVELS},
        "operating_point_gate": {
            "required": True,
            "stages": [
                "durable_candidate",
                "bounded_dry_run",
                "shadow_parse_or_scope_freeze",
                "shadow_vector_and_four_layer_build",
                "measured_evaluation",
                "atomic_promotion",
            ],
            "hard_gates": [
                "base_state_unchanged",
                "qdrant_schema_match",
                "vector_record_coverage",
                "structure_recovery",
                "retrieval_quality",
                "citation_quality",
                "latency_budget",
                "resource_budget",
            ],
        },
        "redaction": {
            "secret_fields": ["chat_api_key", "graph_api_key", "embedding_api_key", "model_bridge_admin_token", "api_keys"],
            "payload_exposes_secret_values": False,
        },
    }


def model_settings_payload(
    *,
    include_dynamic_status: bool = True,
    include_audit_status: bool = True,
) -> dict:
    active_settings = get_settings()
    env_entries = {} if runtime_settings_override_active() else _env_entries(ENV_PATH)
    settings = _settings_from_root_env(active_settings)
    chat_base_url = env_entries.get("CHAT_BASE_URL", settings.chat_base_url)
    graph_base_url = env_entries.get("GRAPH_BASE_URL", settings.graph_base_url)
    embedding_base_url = env_entries.get("EMBEDDING_BASE_URL", settings.embedding_base_url)
    model_bridge_enabled = settings.model_bridge_enabled
    model_bridge_status = (
        model_bridge_status_payload(settings=active_settings, env_entries=env_entries)
        if include_dynamic_status
        else None
    )
    payload = {
        "provider": "multi_protocol",
        "chat_api_protocol": settings.chat_api_protocol,
        "graph_api_protocol": settings.graph_api_protocol,
        "embedding_api_protocol": settings.embedding_api_protocol,
        "chat_base_url": chat_base_url,
        "graph_base_url": graph_base_url,
        "embedding_base_url": embedding_base_url,
        "effective_chat_base_url": active_settings.chat_base_url,
        "effective_graph_base_url": active_settings.graph_base_url,
        "effective_embedding_base_url": active_settings.embedding_base_url,
        "model_bridge_enabled": model_bridge_enabled,
        "chat_resolve_ip": env_entries.get("CHAT_RESOLVE_IP", "" if settings.model_bridge_enabled else (settings.chat_resolve_ip or "")),
        "graph_resolve_ip": env_entries.get("GRAPH_RESOLVE_IP", settings.graph_resolve_ip or ""),
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model,
        "graph_model": settings.graph_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "embedding_batch_size": read_env_int("EMBEDDING_BATCH_SIZE", settings.embedding_batch_size),
        "worker_concurrency": settings.worker_concurrency,
        "model_request_concurrency": settings.model_request_concurrency,
        "model_request_timeout_seconds": settings.model_request_timeout_seconds,
        "chat_json_max_tokens": settings.chat_json_max_tokens,
        "agent_request_concurrency": read_env_int("AGENT_REQUEST_CONCURRENCY", settings.agent_request_concurrency),
        "source_io_concurrency": read_env_int(
            "SOURCE_IO_CONCURRENCY", settings.source_io_concurrency
        ),
        "agent_request_queue_limit": read_env_int("AGENT_REQUEST_QUEUE_LIMIT", settings.agent_request_queue_limit),
        "agent_request_queue_timeout_seconds": read_env_int(
            "AGENT_REQUEST_QUEUE_TIMEOUT_SECONDS", settings.agent_request_queue_timeout_seconds
        ),
        "agent_request_lease_ttl_seconds": read_env_int(
            "AGENT_REQUEST_LEASE_TTL_SECONDS", settings.agent_request_lease_ttl_seconds
        ),
        "upload_max_bytes": read_env_int("UPLOAD_MAX_BYTES", settings.upload_max_bytes),
        "fixed_chunk_size_tokens": read_env_int("FIXED_CHUNK_SIZE_TOKENS", settings.fixed_chunk_size_tokens),
        "fixed_chunk_overlap_tokens": read_env_int("FIXED_CHUNK_OVERLAP_TOKENS", settings.fixed_chunk_overlap_tokens),
        "context_package_token_budget": read_env_int("CONTEXT_PACKAGE_TOKEN_BUDGET", settings.context_package_token_budget),
        "mid_concept_extraction_max_model_batches": read_env_int(
            "MID_CONCEPT_EXTRACTION_MAX_MODEL_BATCHES", settings.mid_concept_extraction_max_model_batches
        ),
        "mid_concept_extraction_max_candidates_per_batch": read_env_int(
            "MID_CONCEPT_EXTRACTION_MAX_CANDIDATES_PER_BATCH", settings.mid_concept_extraction_max_candidates_per_batch
        ),
        "mid_concept_extraction_max_tokens_per_batch": read_env_int(
            "MID_CONCEPT_EXTRACTION_MAX_TOKENS_PER_BATCH", settings.mid_concept_extraction_max_tokens_per_batch
        ),
        "mid_concept_candidate_keep_threshold": read_env_float("MID_CONCEPT_CANDIDATE_KEEP_THRESHOLD", settings.mid_concept_candidate_keep_threshold),
        "rq_kmeans_levels": ACTIVE_RQ_KMEANS_LEVELS,
        "edge_distance_protocol": settings.edge_distance_protocol,
        "rq_membership_protocol": settings.rq_membership_protocol,
        "edge_projection_protocol": settings.edge_projection_protocol,
        "edge_type_calibration_protocol": settings.edge_type_calibration_protocol,
        "rq_kmeans_max_k": read_env_int("RQ_KMEANS_MAX_K", settings.rq_kmeans_max_k),
        "rq_residual_tau": read_env_float("RQ_RESIDUAL_TAU", settings.rq_residual_tau),
        "rq_membership_temperature": settings.rq_membership_temperature,
        "rq_membership_top_m": settings.rq_membership_top_m,
        "rq_membership_probability_threshold": settings.rq_membership_probability_threshold,
        "dense_knn_k_min": read_env_int("DENSE_KNN_K_MIN", settings.dense_knn_k_min),
        "dense_knn_k_max": read_env_int("DENSE_KNN_K_MAX", settings.dense_knn_k_max),
        "dense_reverse_b_min_base": read_env_int("DENSE_REVERSE_B_MIN_BASE", settings.dense_reverse_b_min_base),
        "dense_reverse_b_max_base": read_env_int("DENSE_REVERSE_B_MAX_BASE", settings.dense_reverse_b_max_base),
        "dense_reverse_b_min_doc": read_env_int("DENSE_REVERSE_B_MIN_DOC", settings.dense_reverse_b_min_doc),
        "dense_reverse_b_max_doc": read_env_int("DENSE_REVERSE_B_MAX_DOC", settings.dense_reverse_b_max_doc),
        "dense_reverse_b_min_lang": read_env_int("DENSE_REVERSE_B_MIN_LANG", settings.dense_reverse_b_min_lang),
        "dense_reverse_b_max_lang": read_env_int("DENSE_REVERSE_B_MAX_LANG", settings.dense_reverse_b_max_lang),
        "dense_min_cosine": read_env_float("DENSE_MIN_COSINE", settings.dense_min_cosine),
        "dense_strong_cosine": read_env_float("DENSE_STRONG_COSINE", settings.dense_strong_cosine),
        "cross_doc_out_quota_min": read_env_int("CROSS_DOC_OUT_QUOTA_MIN", settings.cross_doc_out_quota_min),
        "cross_doc_out_quota_max": read_env_int("CROSS_DOC_OUT_QUOTA_MAX", settings.cross_doc_out_quota_max),
        "cross_doc_min_cosine": read_env_float("CROSS_DOC_MIN_COSINE", settings.cross_doc_min_cosine),
        "cross_language_out_quota_min": read_env_int("CROSS_LANGUAGE_OUT_QUOTA_MIN", settings.cross_language_out_quota_min),
        "cross_language_out_quota_max": read_env_int("CROSS_LANGUAGE_OUT_QUOTA_MAX", settings.cross_language_out_quota_max),
        "cross_language_min_cosine": read_env_float("CROSS_LANGUAGE_MIN_COSINE", settings.cross_language_min_cosine),
        "enable_auto_tpe": read_env_bool("ENABLE_AUTO_TPE", settings.enable_auto_tpe),
        "tpe_trial_budget": read_env_int("TPE_TRIAL_BUDGET", settings.tpe_trial_budget),
        "tpe_startup_random_trials": read_env_int("TPE_STARTUP_RANDOM_TRIALS", settings.tpe_startup_random_trials),
        "tpe_good_quantile_gamma": read_env_float("TPE_GOOD_QUANTILE_GAMMA", settings.tpe_good_quantile_gamma),
        "tpe_probe_query_budget": read_env_int("TPE_PROBE_QUERY_BUDGET", settings.tpe_probe_query_budget),
        "tpe_trial_timeout_seconds": read_env_int("TPE_TRIAL_TIMEOUT_SECONDS", settings.tpe_trial_timeout_seconds),
        "tpe_candidate_pool_size": read_env_int("TPE_CANDIDATE_POOL_SIZE", settings.tpe_candidate_pool_size),
        "operating_point_hard_gate_max_edge_density": read_env_float(
            "OPERATING_POINT_HARD_GATE_MAX_EDGE_DENSITY", settings.operating_point_hard_gate_max_edge_density
        ),
        "operating_point_hard_gate_max_isolated_ratio": read_env_float(
            "OPERATING_POINT_HARD_GATE_MAX_ISOLATED_RATIO", settings.operating_point_hard_gate_max_isolated_ratio
        ),
        "operating_point_hard_gate_max_hubness_ratio": read_env_float(
            "OPERATING_POINT_HARD_GATE_MAX_HUBNESS_RATIO", settings.operating_point_hard_gate_max_hubness_ratio
        ),
        "operating_point_hard_gate_min_structure_recovery_rate": read_env_float(
            "OPERATING_POINT_HARD_GATE_MIN_STRUCTURE_RECOVERY_RATE", settings.operating_point_hard_gate_min_structure_recovery_rate
        ),
        "operating_point_hard_gate_max_candidate_latency_p95_ms": read_env_int(
            "OPERATING_POINT_HARD_GATE_MAX_CANDIDATE_LATENCY_P95_MS", settings.operating_point_hard_gate_max_candidate_latency_p95_ms
        ),
        "retrieval_result_top_k_default": read_env_int("RETRIEVAL_RESULT_TOP_K_DEFAULT", settings.retrieval_result_top_k_default),
        "agent_coarse_initial_budget": read_env_int(
            "AGENT_COARSE_INITIAL_BUDGET",
            read_env_int("AGENT_COARSE_TOTAL_BUDGET", settings.agent_coarse_initial_budget or settings.agent_coarse_total_budget),
        ),
        "agent_coarse_total_budget": read_env_int("AGENT_COARSE_TOTAL_BUDGET", settings.agent_coarse_total_budget),
        "agent_coarse_top_k": read_env_int(
            "AGENT_COARSE_TOP_K",
            settings.agent_coarse_top_k or settings.agent_coarse_initial_budget or settings.agent_coarse_total_budget,
        ),
        "agent_mid_per_coarse_budget": read_env_int("AGENT_MID_PER_COARSE_BUDGET", settings.agent_mid_per_coarse_budget),
        "agent_coarse_drilldown_mid_initial_budget": read_env_int(
            "AGENT_COARSE_DRILLDOWN_MID_INITIAL_BUDGET",
            settings.agent_coarse_drilldown_mid_initial_budget or settings.agent_mid_top_k,
        ),
        "agent_mid_initial_budget": read_env_int("AGENT_MID_INITIAL_BUDGET", settings.agent_mid_initial_budget or settings.agent_mid_top_k),
        "agent_mid_top_k": read_env_int("AGENT_MID_TOP_K", settings.agent_mid_top_k),
        "agent_chunk_per_mid_budget": read_env_int("AGENT_CHUNK_PER_MID_BUDGET", settings.agent_chunk_per_mid_budget),
        "agent_chunk_initial_budget": read_env_int("AGENT_CHUNK_INITIAL_BUDGET", settings.agent_chunk_initial_budget or settings.agent_chunk_top_k),
        "agent_chunk_top_k": read_env_int("AGENT_CHUNK_TOP_K", settings.agent_chunk_top_k),
        "agent_max_depth_per_layer": read_env_int("AGENT_MAX_DEPTH_PER_LAYER", settings.agent_max_depth_per_layer),
        "agent_max_labels_per_node": read_env_int("AGENT_MAX_LABELS_PER_NODE", settings.agent_max_labels_per_node),
        "agent_max_edge_reuse": read_env_int("AGENT_MAX_EDGE_REUSE", settings.agent_max_edge_reuse),
        "agent_max_cycle_reward_per_path": read_env_float("AGENT_MAX_CYCLE_REWARD_PER_PATH", settings.agent_max_cycle_reward_per_path),
        "agent_cycle_reward_distance_threshold": read_env_float("AGENT_CYCLE_REWARD_DISTANCE_THRESHOLD", settings.agent_cycle_reward_distance_threshold),
        "agent_path_distance_green_threshold": read_env_float("AGENT_PATH_DISTANCE_GREEN_THRESHOLD", settings.agent_path_distance_green_threshold),
        "agent_path_distance_gray_threshold": read_env_float("AGENT_PATH_DISTANCE_GRAY_THRESHOLD", settings.agent_path_distance_gray_threshold),
        "agent_path_distance_hard_threshold": read_env_float("AGENT_PATH_DISTANCE_HARD_THRESHOLD", settings.agent_path_distance_hard_threshold),
        "gray_zone_rule_protocol": env_entries.get("GRAY_ZONE_RULE_PROTOCOL", settings.gray_zone_rule_protocol),
        "gray_zone_observation_cadence": read_env_int(
            "GRAY_ZONE_OBSERVATION_CADENCE",
            settings.gray_zone_observation_cadence,
        ),
        "traversal_observation_budget": read_env_int(
            "TRAVERSAL_OBSERVATION_BUDGET",
            settings.traversal_observation_budget,
        ),
        "candidate_pool_dedupe_budget": read_env_int("CANDIDATE_POOL_DEDUPE_BUDGET", settings.candidate_pool_dedupe_budget),
        "agent_structure_restore_per_chunk_budget": read_env_int(
            "AGENT_STRUCTURE_RESTORE_PER_CHUNK_BUDGET",
            settings.agent_structure_restore_per_chunk_budget or settings.agent_structure_restore_budget,
        ),
        "agent_structure_restore_budget": read_env_int("AGENT_STRUCTURE_RESTORE_BUDGET", settings.agent_structure_restore_budget),
        "context_path_summary_budget": read_env_int("CONTEXT_PATH_SUMMARY_BUDGET", settings.context_path_summary_budget),
        "agent_planning_round_budget": read_env_int("AGENT_PLANNING_ROUND_BUDGET", settings.agent_planning_round_budget),
        "agent_max_typed_actions_per_round": read_env_int("AGENT_MAX_TYPED_ACTIONS_PER_ROUND", settings.agent_max_typed_actions_per_round),
        "agent_repair_round_budget": read_env_int("AGENT_REPAIR_ROUND_BUDGET", settings.agent_repair_round_budget),
        "agent_verification_budget": read_env_int("AGENT_VERIFICATION_BUDGET", settings.agent_verification_budget),
        "concept_i18n_enabled": read_env_bool("CONCEPT_I18N_ENABLED", settings.concept_i18n_enabled),
        "query_facet_bilingual_enabled": read_env_bool("QUERY_FACET_BILINGUAL_ENABLED", settings.query_facet_bilingual_enabled),
        "query_facet_posterior_enabled": read_env_bool(
            "QUERY_FACET_POSTERIOR_ENABLED",
            settings.query_facet_posterior_enabled,
        ),
        "query_facet_posterior_observation_budget": read_env_int(
            "QUERY_FACET_POSTERIOR_OBSERVATION_BUDGET",
            settings.query_facet_posterior_observation_budget,
        ),
        "query_facet_posterior_round_budget": read_env_int(
            "QUERY_FACET_POSTERIOR_ROUND_BUDGET",
            settings.query_facet_posterior_round_budget,
        ),
        "query_facet_posterior_convergence_epsilon": read_env_float(
            "QUERY_FACET_POSTERIOR_CONVERGENCE_EPSILON",
            settings.query_facet_posterior_convergence_epsilon,
        ),
        "enable_model_fallback": settings.enable_model_fallback,
        "enable_database_fallback": settings.enable_database_fallback,
        "has_chat_api_key": bool(settings.chat_api_key),
        "has_graph_api_key": bool(settings.graph_api_key),
        "degraded_mode": not settings.chat_api_key or not settings.embedding_api_key or not settings.embedding_base_url,
        "embedding_resolve_ip": env_entries.get("EMBEDDING_RESOLVE_IP", "" if settings.model_bridge_enabled else (settings.embedding_resolve_ip or "")),
        "has_embedding_api_key": bool(settings.embedding_api_key),
        "model_bridge_status": model_bridge_status,
        "runtime_settings_version": (
            current_runtime_settings_version() if include_dynamic_status else None
        ),
        "lifecycle": runtime_lifecycle_payload(),
    }
    if not include_dynamic_status:
        payload.pop("model_bridge_status", None)
        payload.pop("runtime_settings_version", None)
    if include_dynamic_status and include_audit_status:
        payload.update(runtime_settings_audit_status_payload(configured_payload=payload))
    return payload


class RuntimeEnvFileConflict(RuntimeError):
    """The shared env file no longer matches an exact confirmed identity."""


class RuntimeEnvFileLockTimeout(RuntimeError):
    """The shared env writer fence could not be acquired within its bound."""


class RuntimeEnvFileRecoveryRequired(RuntimeError):
    """A durable env publication receipt requires an explicit resolution."""

    def __init__(
        self,
        message: str,
        *,
        recovery_fact: Mapping[str, Any] | None = None,
        recovery_id: str | None = None,
    ):
        self.recovery_fact = dict(recovery_fact or {})
        self.recovery_id = str(
            recovery_id or self.recovery_fact.get("recovery_id") or ""
        ) or None
        rendered = str(message)
        if self.recovery_id and self.recovery_id not in rendered:
            rendered = f"{rendered}; recovery_id={self.recovery_id}"
        super().__init__(rendered)


class RuntimeEnvFilePublishError(OSError):
    """An env publication failed after recording an auditable recovery fact."""

    def __init__(self, message: str, *, recovery_fact: Mapping[str, Any]):
        super().__init__(message)
        self.recovery_fact = dict(recovery_fact)


@dataclass(frozen=True)
class RuntimeEnvFileMutationReceipt:
    """In-memory rollback capability; raw env bytes are never serialized."""

    protocol_version: str
    target_path: str
    changed: bool
    before_identity: dict[str, Any]
    after_identity: dict[str, Any]
    before_bytes: bytes | None = field(repr=False)
    recovery_id: str | None = None
    recovery_journal_path_hash: str | None = None


# Recovery metadata is deliberately published through separately captured
# primitives.  Target-file fsync/replace remain patchable fault-injection
# points, while recovery-journal and compensation faults can be injected by
# patching these named callables independently.
_RUNTIME_ENV_RECOVERY_FSYNC = os.fsync
_RUNTIME_ENV_OS_REPLACE = os.replace


def _windows_move_file_write_through(source: Path, target: Path) -> None:
    """Replace ``target`` with a write-through same-volume Windows rename."""

    import ctypes

    move_file_ex = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    move_file_ex.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    move_file_ex.restype = ctypes.c_int
    movefile_replace_existing = 0x00000001
    movefile_write_through = 0x00000008
    if not move_file_ex(
        str(source),
        str(target),
        movefile_replace_existing | movefile_write_through,
    ):
        error_code = ctypes.get_last_error()
        raise OSError(
            error_code,
            ctypes.FormatError(error_code),
            str(target),
        )


def _runtime_env_atomic_replace(source: Path, target: Path) -> None:
    """Atomic namespace replacement with a platform durability primitive."""

    if os.name == "nt":
        _windows_move_file_write_through(Path(source), Path(target))
        return
    _RUNTIME_ENV_OS_REPLACE(source, target)


def _runtime_env_target_replace(source: Path, target: Path) -> None:
    """Target replace hook retaining explicit ``os.replace`` fault injection."""

    if os.replace is not _RUNTIME_ENV_OS_REPLACE:
        os.replace(source, target)
        return
    _runtime_env_atomic_replace(source, target)


_RUNTIME_ENV_RECOVERY_REPLACE = _runtime_env_atomic_replace


def _runtime_env_target_path(path: Path | str | None = None) -> Path:
    candidate = Path(path) if path is not None else Path(ENV_PATH)
    return Path(os.path.abspath(os.path.normpath(str(candidate))))


def _runtime_env_identity_hash(identity: Mapping[str, Any]) -> str:
    payload = {
        str(key): value
        for key, value in identity.items()
        if str(key) != "identity_hash"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_runtime_env_snapshot(
    path: Path | str | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    target = _runtime_env_target_path(path)
    base_identity: dict[str, Any] = {
        "protocol_version": RUNTIME_ENV_FILE_IDENTITY_PROTOCOL_VERSION,
        "path": str(target),
    }
    try:
        before = os.lstat(target)
    except FileNotFoundError:
        identity = {
            **base_identity,
            "exists": False,
            "device_id": None,
            "inode": None,
            "size_bytes": 0,
            "mtime_ns": None,
            "ctime_ns": None,
            "file_sha256": None,
        }
        identity["identity_hash"] = _runtime_env_identity_hash(identity)
        return identity, None
    if stat.S_ISLNK(before.st_mode):
        raise RuntimeEnvFileConflict(
            "Shared runtime env target must not be a symbolic link"
        )
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeEnvFileConflict(
            "Shared runtime env target must be a regular file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            int(opened.st_dev) != int(before.st_dev)
            or int(opened.st_ino) != int(before.st_ino)
        ):
            raise RuntimeEnvFileConflict(
                "Shared runtime env identity changed while it was opened"
            )
        digest = hashlib.sha256()
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 64 * 1024)
            if not block:
                break
            digest.update(block)
            blocks.append(block)
        content = b"".join(blocks)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = os.lstat(target)
    except FileNotFoundError as exc:
        raise RuntimeEnvFileConflict(
            "Shared runtime env target disappeared during identity capture"
        ) from exc
    # Windows exposes different ctime semantics through a descriptor fstat
    # and a path lstat for the same file (creation time versus metadata-change
    # time).  Device/inode, size, mtime and the streamed SHA-256 are the stable
    # cross-platform replacement/mutation proof.  Keep path ctime in the
    # identity card for audit, but never compare it across the two APIs.
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
    )
    if any(
        int(getattr(opened, field_name))
        != int(getattr(after_open, field_name))
        or int(getattr(opened, field_name))
        != int(getattr(after_path, field_name))
        for field_name in stable_fields
    ):
        raise RuntimeEnvFileConflict(
            "Shared runtime env bytes changed during identity capture"
        )
    identity = {
        **base_identity,
        "exists": True,
        "device_id": int(opened.st_dev),
        "inode": int(opened.st_ino),
        "size_bytes": int(opened.st_size),
        "mtime_ns": int(opened.st_mtime_ns),
        "ctime_ns": int(after_path.st_ctime_ns),
        "file_sha256": digest.hexdigest(),
    }
    identity["identity_hash"] = _runtime_env_identity_hash(identity)
    return identity, content


def runtime_env_file_identity(
    path: Path | str | None = None,
) -> dict[str, Any]:
    identity, _content = _read_runtime_env_snapshot(path)
    return identity


def _runtime_env_initial_activation_backup_path(intent_id: str) -> Path:
    try:
        canonical_id = str(UUID(str(intent_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Initial graph activation intent id is invalid") from exc
    target = _runtime_env_target_path(ENV_PATH)
    return target.with_name(
        f"{target.name}.symbograph.initial-graph-activation.{canonical_id}.bin"
    )


def write_runtime_env_initial_activation_backup(
    *,
    intent_id: str,
    expected_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist an exact mode-0600 compensation backup without exposing bytes."""

    with runtime_env_file_lock(path=ENV_PATH):
        _require_runtime_env_recovery_resolved(ENV_PATH)
        source_identity, source_bytes = _require_runtime_env_identity(
            expected_source_identity,
            path=ENV_PATH,
        )
        if source_bytes is None:
            raise RuntimeEnvFileConflict(
                "Initial graph activation backup requires an existing env file"
            )
        backup_path = _runtime_env_initial_activation_backup_path(intent_id)
        observed_identity, observed_bytes = _read_runtime_env_snapshot(backup_path)
        if observed_bytes is not None:
            if observed_bytes != source_bytes:
                raise RuntimeEnvFileConflict(
                    "Initial graph activation backup already exists with different bytes"
                )
        else:
            _write_runtime_env_recovery_control_file(backup_path, source_bytes)
            with suppress(OSError):
                os.chmod(backup_path, 0o600)
        backup_identity = runtime_env_file_identity(backup_path)
        if (
            backup_identity.get("file_sha256")
            != source_identity.get("file_sha256")
            or backup_identity.get("size_bytes")
            != source_identity.get("size_bytes")
        ):
            raise RuntimeEnvFileConflict(
                "Initial graph activation backup did not preserve exact bytes"
            )
        return {
            "protocol_version": "runtime_env_initial_activation_backup_v1",
            "path_hash": _runtime_env_recovery_path_hash(backup_path),
            "file_sha256": backup_identity["file_sha256"],
            "size_bytes": backup_identity["size_bytes"],
            "identity_hash": backup_identity["identity_hash"],
        }


def read_runtime_env_initial_activation_backup(
    *,
    intent_id: str,
    expected_card: Mapping[str, Any],
) -> bytes:
    required = {
        "protocol_version",
        "path_hash",
        "file_sha256",
        "size_bytes",
        "identity_hash",
    }
    if not isinstance(expected_card, Mapping) or set(expected_card) != required:
        raise RuntimeEnvFileConflict(
            "Initial graph activation backup card has an invalid schema"
        )
    backup_path = _runtime_env_initial_activation_backup_path(intent_id)
    if expected_card.get("path_hash") != _runtime_env_recovery_path_hash(
        backup_path
    ):
        raise RuntimeEnvFileConflict(
            "Initial graph activation backup path hash drifted"
        )
    identity, content = _read_runtime_env_snapshot(backup_path)
    if content is None:
        raise RuntimeEnvFileRecoveryRequired(
            "Initial graph activation compensation backup is missing"
        )
    if (
        expected_card.get("protocol_version")
        != "runtime_env_initial_activation_backup_v1"
        or expected_card.get("file_sha256") != identity.get("file_sha256")
        or expected_card.get("size_bytes") != identity.get("size_bytes")
        or expected_card.get("identity_hash") != identity.get("identity_hash")
    ):
        raise RuntimeEnvFileConflict(
            "Initial graph activation compensation backup identity drifted"
        )
    return content


def remove_runtime_env_initial_activation_backup(
    *,
    intent_id: str,
    expected_card: Mapping[str, Any],
) -> bool:
    # Verify exact bytes/identity before deleting the secret-bearing backup.
    read_runtime_env_initial_activation_backup(
        intent_id=intent_id,
        expected_card=expected_card,
    )
    with runtime_env_file_lock(path=ENV_PATH):
        backup_path = _runtime_env_initial_activation_backup_path(intent_id)
        return _durable_unlink_runtime_env_path(backup_path)


def _runtime_env_recovery_paths(
    path: Path | str | None = None,
) -> tuple[Path, Path, Path]:
    target = _runtime_env_target_path(path)
    return (
        target.with_name(f"{target.name}.symbograph.recovery.json"),
        target.with_name(f"{target.name}.symbograph.recovery.bin"),
        target.with_name(f"{target.name}.symbograph.recovery-audit.json"),
    )


def _runtime_env_recovery_path_hash(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _fsync_runtime_env_recovery_directory(directory: Path) -> str:
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _RUNTIME_ENV_RECOVERY_FSYNC(descriptor)
        finally:
            os.close(descriptor)
        return "posix_parent_directory_fsync_v1"
    if os.name == "nt":
        # Every recovery control-file replacement on Windows is already issued
        # through MoveFileExW(REPLACE_EXISTING | WRITE_THROUGH).  There is no
        # Python directory-fd fsync equivalent, so make that platform contract
        # explicit instead of silently pretending a directory barrier ran.
        return "windows_movefileex_write_through_v1"
    raise OSError(
        "Runtime env recovery namespace durability is unsupported on this platform"
    )


def _durable_unlink_runtime_env_path(path: Path) -> bool:
    """Durably remove an authoritative recovery name from its namespace."""

    try:
        observed = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise RuntimeEnvFileConflict(
            "Runtime env recovery cleanup target is not a regular file"
        )
    if hasattr(os, "O_DIRECTORY"):
        path.unlink()
        _fsync_runtime_env_recovery_directory(path.parent)
        return True
    if os.name != "nt":
        raise OSError(
            "Runtime env recovery unlink durability is unsupported on this platform"
        )
    tombstone = path.with_name(
        f"{path.name}.symbograph.{secrets.token_hex(8)}.tombstone"
    )
    _runtime_env_atomic_replace(path, tombstone)
    # The authoritative name removal is persisted by the write-through move.
    # Physical tombstone deletion is non-authoritative; a later writer also
    # sweeps any crash residue while holding the same cross-process lock.
    tombstone.unlink()
    return True


def _write_runtime_env_recovery_control_file(
    path: Path,
    content: bytes,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".control",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            _RUNTIME_ENV_RECOVERY_FSYNC(handle.fileno())
        _RUNTIME_ENV_RECOVERY_REPLACE(temporary, path)
        replaced = True
        _fsync_runtime_env_recovery_directory(path.parent)
    finally:
        if not replaced:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _write_runtime_env_recovery_state(state: Mapping[str, Any]) -> None:
    journal_path, _backup_path, _audit_path = _runtime_env_recovery_paths(
        str(state["target_path"])
    )
    encoded = (
        json.dumps(
            dict(state),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_runtime_env_recovery_control_file(journal_path, encoded)


def _validated_runtime_env_recovery_state(
    payload: Any,
    *,
    path: Path | str | None = None,
) -> dict[str, Any]:
    target = _runtime_env_target_path(path)
    _journal_path, backup_path, _audit_path = _runtime_env_recovery_paths(
        target
    )
    required_fields = {
        "protocol_version",
        "recovery_id",
        "target_path",
        "target_path_hash",
        "backup_name",
        "before_exists",
        "before_file_sha256",
        "before_size_bytes",
        "publish_file_sha256",
        "publish_size_bytes",
        "commit_policy",
        "phase",
        "after_identity_hash",
        "resolution",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env recovery journal has an invalid frozen schema"
        )
    state = {str(key): value for key, value in payload.items()}
    recovery_id = str(state["recovery_id"])
    valid_recovery_id = (
        len(recovery_id) == 32
        and all(character in "0123456789abcdef" for character in recovery_id)
    )
    valid_phase = state["phase"] in {
        "prepared",
        "published_pending_commit",
        "committed",
        "rolled_back",
    }
    valid_commit_policy = state["commit_policy"] in {
        "caller_commit_required",
        "publish_is_commit",
    }
    if (
        state["protocol_version"]
        != RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION
        or not valid_recovery_id
        or state["target_path"] != str(target)
        or state["target_path_hash"]
        != _runtime_env_recovery_path_hash(target)
        or state["backup_name"] != backup_path.name
        or type(state["before_exists"]) is not bool
        or not valid_phase
        or not valid_commit_policy
    ):
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env recovery journal identity is invalid"
        )
    return state


def _load_runtime_env_recovery_state(
    path: Path | str | None = None,
) -> dict[str, Any] | None:
    target = _runtime_env_target_path(path)
    journal_path, _backup_path, _audit_path = _runtime_env_recovery_paths(
        target
    )
    _identity, content = _read_runtime_env_snapshot(journal_path)
    if content is None:
        return None
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env recovery journal is not valid canonical JSON"
        ) from exc
    return _validated_runtime_env_recovery_state(payload, path=target)


def _runtime_env_recovery_target_position(
    state: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    identity = runtime_env_file_identity(str(state["target_path"]))
    if (
        identity.get("exists") == state.get("before_exists")
        and identity.get("file_sha256") == state.get("before_file_sha256")
        and identity.get("size_bytes") == state.get("before_size_bytes")
    ):
        return "before", identity
    if (
        identity.get("exists") is True
        and identity.get("file_sha256")
        == state.get("publish_file_sha256")
        and identity.get("size_bytes") == state.get("publish_size_bytes")
    ):
        return "published", identity
    return "conflict", identity


def _runtime_env_recovery_fact(
    state: Mapping[str, Any],
    *,
    outcome: str,
) -> dict[str, Any]:
    _journal_path, _backup_path, audit_path = _runtime_env_recovery_paths(
        str(state["target_path"])
    )
    return {
        "protocol_version": RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION,
        "recovery_id": state["recovery_id"],
        "target_path_hash": state["target_path_hash"],
        "outcome": outcome,
        "audit_path_hash": _runtime_env_recovery_path_hash(audit_path),
    }


def _record_runtime_env_recovery_audit(
    state: Mapping[str, Any],
    *,
    outcome: str,
    error_type: str | None,
) -> dict[str, Any]:
    _journal_path, _backup_path, audit_path = _runtime_env_recovery_paths(
        str(state["target_path"])
    )
    fact = _runtime_env_recovery_fact(state, outcome=outcome)
    payload = {
        **fact,
        "error_type": error_type,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_runtime_env_recovery_control_file(audit_path, encoded)
    return fact


def _cleanup_runtime_env_recovery_files(
    state: Mapping[str, Any],
) -> None:
    journal_path, backup_path, _audit_path = _runtime_env_recovery_paths(
        str(state["target_path"])
    )
    for candidate in (backup_path, journal_path):
        _durable_unlink_runtime_env_path(candidate)


def _prepare_runtime_env_recovery(
    *,
    before_identity: Mapping[str, Any],
    before_bytes: bytes | None,
    publish_bytes: bytes,
    commit_policy: str,
    path: Path | str | None = None,
) -> dict[str, Any]:
    target = _runtime_env_target_path(path)
    journal_path, backup_path, _audit_path = _runtime_env_recovery_paths(
        target
    )
    if journal_path.exists():
        pending = _load_runtime_env_recovery_state(target)
        recovery_id = pending.get("recovery_id") if pending else "unknown"
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env has an unresolved durable recovery receipt; "
            f"recovery_id={recovery_id}"
        )
    _durable_unlink_runtime_env_path(backup_path)
    state = {
        "protocol_version": RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION,
        "recovery_id": secrets.token_hex(16),
        "target_path": str(target),
        "target_path_hash": _runtime_env_recovery_path_hash(target),
        "backup_name": backup_path.name,
        "before_exists": bool(before_identity.get("exists")),
        "before_file_sha256": before_identity.get("file_sha256"),
        "before_size_bytes": before_identity.get("size_bytes"),
        "publish_file_sha256": hashlib.sha256(publish_bytes).hexdigest(),
        "publish_size_bytes": len(publish_bytes),
        "commit_policy": commit_policy,
        "phase": "prepared",
        "after_identity_hash": None,
        "resolution": None,
    }
    try:
        _write_runtime_env_recovery_control_file(
            backup_path,
            before_bytes or b"",
        )
        _write_runtime_env_recovery_state(state)
    except BaseException as preparation_error:
        position, _identity = _runtime_env_recovery_target_position(state)
        if position != "before":
            raise RuntimeEnvFileRecoveryRequired(
                "Runtime env recovery preparation failed and target identity "
                f"is ambiguous; recovery_id={state['recovery_id']}; "
                f"preparation_error_type={type(preparation_error).__name__}"
            ) from preparation_error
        fact = _runtime_env_recovery_fact(
            state,
            outcome="preparation_failed_target_unchanged",
        )
        audit_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            _record_runtime_env_recovery_audit(
                state,
                outcome="preparation_failed_target_unchanged",
                error_type=type(preparation_error).__name__,
            )
        except BaseException as exc:
            audit_error = exc
        try:
            _cleanup_runtime_env_recovery_files(state)
        except BaseException as exc:
            cleanup_error = exc
        raise RuntimeEnvFilePublishError(
            "Runtime env recovery preparation failed before target mutation; "
            f"recovery_id={state['recovery_id']}; "
            f"preparation_error_type={type(preparation_error).__name__}; "
            f"audit_error_type={type(audit_error).__name__ if audit_error else 'none'}; "
            f"cleanup_error_type={type(cleanup_error).__name__ if cleanup_error else 'none'}",
            recovery_fact={
                **fact,
                "target_unchanged": True,
                "audit_persisted": audit_error is None,
                "cleanup_completed": cleanup_error is None,
            },
        ) from preparation_error
    return state


def _read_runtime_env_recovery_backup(
    state: Mapping[str, Any],
) -> bytes:
    _journal_path, backup_path, _audit_path = _runtime_env_recovery_paths(
        str(state["target_path"])
    )
    _identity, backup_bytes = _read_runtime_env_snapshot(backup_path)
    backup_matches = (
        backup_bytes == b""
        if not state.get("before_exists")
        else (
            backup_bytes is not None
            and hashlib.sha256(backup_bytes).hexdigest()
            == state.get("before_file_sha256")
            and len(backup_bytes) == state.get("before_size_bytes")
        )
    )
    if not backup_matches:
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env recovery backup is missing or does not match its receipt"
        )
    return backup_bytes


def runtime_env_file_recovery_status(
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Return secret-free recovery state for health/maintenance diagnostics."""

    target = _runtime_env_target_path(path)
    state = _load_runtime_env_recovery_state(target)
    if state is None:
        return {
            "protocol_version": RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION,
            "pending": False,
            "target_path_hash": _runtime_env_recovery_path_hash(target),
        }
    position, _identity = _runtime_env_recovery_target_position(state)
    return {
        "protocol_version": RUNTIME_ENV_FILE_RECOVERY_PROTOCOL_VERSION,
        "pending": state["phase"] not in {"committed", "rolled_back"},
        "recovery_id": state["recovery_id"],
        "target_path_hash": state["target_path_hash"],
        "phase": state["phase"],
        "target_position": position,
    }


def _require_runtime_env_recovery_resolved(
    path: Path | str | None = None,
) -> None:
    target = _runtime_env_target_path(path)
    journal_path, backup_path, _audit_path = _runtime_env_recovery_paths(
        target
    )
    state = _load_runtime_env_recovery_state(target)
    if state is None:
        _durable_unlink_runtime_env_path(backup_path)
        return
    position, _identity = _runtime_env_recovery_target_position(state)
    if (
        (state["phase"] == "committed" and position == "published")
        or (state["phase"] == "rolled_back" and position == "before")
    ):
        _cleanup_runtime_env_recovery_files(state)
        return
    if state["phase"] == "prepared" and position == "before":
        resolved = {
            **state,
            "phase": "rolled_back",
            "resolution": "abandoned_before_target_publish",
        }
        _write_runtime_env_recovery_state(resolved)
        _record_runtime_env_recovery_audit(
            resolved,
            outcome="abandoned_before_target_publish",
            error_type=None,
        )
        _cleanup_runtime_env_recovery_files(resolved)
        return
    if (
        state["commit_policy"] == "publish_is_commit"
        and state["phase"] in {"prepared", "published_pending_commit"}
        and position == "published"
    ):
        committed = {
            **state,
            "phase": "committed",
            "resolution": "complete_auto_commit_recovered_after_crash",
        }
        _write_runtime_env_recovery_state(committed)
        _record_runtime_env_recovery_audit(
            committed,
            outcome="complete_auto_commit_recovered_after_crash",
            error_type=None,
        )
        _cleanup_runtime_env_recovery_files(committed)
        return
    raise RuntimeEnvFileRecoveryRequired(
        "Runtime env durable recovery receipt is unresolved; "
        f"recovery_id={state['recovery_id']}; phase={state['phase']}; "
        f"target_position={position}"
    )


def _validated_expected_runtime_env_identity(
    expected_identity: Mapping[str, Any],
    *,
    path: Path | str | None = None,
) -> dict[str, Any]:
    if not isinstance(expected_identity, Mapping):
        raise RuntimeEnvFileConflict(
            "Expected runtime env identity must be an object"
        )
    expected = {str(key): value for key, value in expected_identity.items()}
    required_fields = {
        "protocol_version",
        "path",
        "exists",
        "device_id",
        "inode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "file_sha256",
        "identity_hash",
    }
    if set(expected) != required_fields:
        raise RuntimeEnvFileConflict(
            "Expected runtime env identity has an invalid frozen schema"
        )
    if (
        expected["protocol_version"]
        != RUNTIME_ENV_FILE_IDENTITY_PROTOCOL_VERSION
        or expected["path"] != str(_runtime_env_target_path(path))
        or type(expected["exists"]) is not bool
        or expected["identity_hash"]
        != _runtime_env_identity_hash(expected)
    ):
        raise RuntimeEnvFileConflict(
            "Expected runtime env identity is invalid"
        )
    return expected


def _require_runtime_env_identity(
    expected_identity: Mapping[str, Any],
    *,
    path: Path | str | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    expected = _validated_expected_runtime_env_identity(
        expected_identity,
        path=path,
    )
    observed, content = _read_runtime_env_snapshot(path)
    if observed != expected:
        raise RuntimeEnvFileConflict(
            "Shared runtime env identity/hash changed after confirmation"
        )
    return observed, content


def require_runtime_env_file_identity(
    expected_identity: Mapping[str, Any],
    *,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate the frozen identity schema and current target without bytes."""

    observed, _content = _require_runtime_env_identity(
        expected_identity,
        path=path,
    )
    return observed


def _runtime_env_lock_path(path: Path | str | None = None) -> Path:
    target = _runtime_env_target_path(path)
    lock_root = Path(tempfile.gettempdir()) / "symbograph-runtime-env-locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    target_hash = hashlib.sha256(str(target).encode("utf-8")).hexdigest()
    return lock_root / f"{target_hash}.lock"


def _cleanup_runtime_env_temporary_files(
    path: Path | str | None = None,
) -> int:
    """Remove only orphaned atomic-write files while holding the writer lock."""

    target = _runtime_env_target_path(path)
    prefix = f"{target.name}.symbograph."
    orphan_suffixes = (".tmp", ".tombstone")
    removed = 0
    with os.scandir(target.parent) as entries:
        for entry in entries:
            if (
                not entry.name.startswith(prefix)
                or not entry.name.endswith(orphan_suffixes)
            ):
                continue
            os.unlink(entry.path)
            removed += 1
    return removed


@contextmanager
def runtime_env_file_lock(
    *,
    path: Path | str | None = None,
    timeout_seconds: float = RUNTIME_ENV_FILE_LOCK_TIMEOUT_SECONDS,
) -> Iterator[dict[str, Any]]:
    """Serialize every shared-env writer across API/worker processes."""

    bounded_timeout = float(timeout_seconds)
    if not 0 < bounded_timeout <= RUNTIME_ENV_FILE_LOCK_TIMEOUT_SECONDS:
        raise ValueError(
            "Runtime env file lock timeout must be greater than zero and no "
            f"more than {RUNTIME_ENV_FILE_LOCK_TIMEOUT_SECONDS}"
        )
    target = _runtime_env_target_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _runtime_env_lock_path(target)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    deadline = time.monotonic() + bounded_timeout
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(handle.fileno()).st_size < 1:
                handle.write(b"\0")
                handle.flush()
            while not acquired:
                try:
                    handle.seek(0)
                    msvcrt.locking(
                        handle.fileno(),
                        msvcrt.LK_NBLCK,
                        1,
                    )
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeEnvFileLockTimeout(
                            "Timed out acquiring the shared runtime env file lock"
                        )
                    time.sleep(RUNTIME_ENV_FILE_LOCK_POLL_SECONDS)
        else:
            import fcntl

            while not acquired:
                try:
                    fcntl.flock(
                        handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise RuntimeEnvFileLockTimeout(
                            "Timed out acquiring the shared runtime env file lock"
                        )
                    time.sleep(RUNTIME_ENV_FILE_LOCK_POLL_SECONDS)
        yield {
            "protocol_version": RUNTIME_ENV_FILE_LOCK_PROTOCOL_VERSION,
            "target_path": str(target),
            "lock_path_hash": hashlib.sha256(
                str(lock_path).encode("utf-8")
            ).hexdigest(),
        }
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _fsync_runtime_env_publish_barrier(
    target: Path,
    *,
    fsync_callable,
    expected_content: bytes,
) -> None:
    """Complete and verify the platform-specific publication barrier.

    POSIX persists the renamed entry by fsyncing its parent directory.  The
    Windows namespace move has already used ``MoveFileExW`` with
    ``MOVEFILE_WRITE_THROUGH``; reopen the resulting file, commit it, replay
    its exact bytes and prove the path still names the opened identity.
    """

    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(
            target.parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
    else:
        before = os.lstat(target)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeEnvFileConflict(
                "Published runtime env target is not a regular file"
            )
        # Windows' CRT rejects ``os.fsync`` on a read-only descriptor even
        # though no bytes are changed.  Open the managed replacement read/write
        # solely so ``_commit``/``FlushFileBuffers`` can provide the barrier.
        flags = os.O_RDWR
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOINHERIT"):
            flags |= os.O_NOINHERIT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(target, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_dev) != int(before.st_dev)
            or int(opened.st_ino) != int(before.st_ino)
        ):
            os.close(descriptor)
            raise RuntimeEnvFileConflict(
                "Published runtime env target changed before durability barrier"
            )
    try:
        fsync_callable(descriptor)
        if not hasattr(os, "O_DIRECTORY"):
            os.lseek(descriptor, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            size_bytes = 0
            while True:
                block = os.read(descriptor, 64 * 1024)
                if not block:
                    break
                digest.update(block)
                size_bytes += len(block)
            after_open = os.fstat(descriptor)
            after_path = os.lstat(target)
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if (
                any(
                    int(getattr(opened, field_name))
                    != int(getattr(after_open, field_name))
                    or int(getattr(opened, field_name))
                    != int(getattr(after_path, field_name))
                    for field_name in stable_fields
                )
                or size_bytes != len(expected_content)
                or digest.hexdigest()
                != hashlib.sha256(expected_content).hexdigest()
            ):
                raise RuntimeEnvFileConflict(
                    "Published runtime env target changed during durability verification"
                )
    finally:
        os.close(descriptor)


def _atomic_replace_runtime_env_bytes(
    content: bytes,
    *,
    path: Path | str | None = None,
    fsync_callable=None,
    replace_callable=None,
) -> None:
    target = _runtime_env_target_path(path)
    file_fsync = fsync_callable or os.fsync
    atomic_replace = replace_callable or _runtime_env_target_replace
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{target.name}.symbograph.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            file_fsync(handle.fileno())
        atomic_replace(temporary, target)
        replaced = True
        _fsync_runtime_env_publish_barrier(
            target,
            fsync_callable=file_fsync,
            expected_content=content,
        )
    finally:
        if not replaced:
            with suppress(FileNotFoundError):
                temporary.unlink()


def _serialize_env_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if not text:
        return ""
    if any(char.isspace() for char in text) or any(char in text for char in ['"', "#", "="]):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _runtime_env_bytes_with_updates(
    content: bytes | None,
    updates: Mapping[str, str | int | float | bool | None],
) -> bytes:
    """Render an exact managed-env update without mutating process or disk."""

    lines = (
        (content or b"").decode("utf-8").splitlines()
        if content is not None
        else []
    )
    remaining = {str(key).upper(): value for key, value in updates.items()}
    next_lines: list[str] = []
    seen_keys: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            next_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip().lstrip("\ufeff").upper()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if key not in remaining:
            value = line.split("=", 1)[1]
            next_lines.append(f"{key}={value}")
            continue
        value = remaining.pop(key)
        if value is not None:
            next_lines.append(f"{key}={_serialize_env_value(value)}")

    if remaining:
        if next_lines and next_lines[-1].strip():
            next_lines.append("")
        for key, value in remaining.items():
            if value is not None:
                next_lines.append(f"{key}={_serialize_env_value(value)}")
    return ("\n".join(next_lines).rstrip() + "\n").encode("utf-8")


def _update_env_file(
    updates: dict[str, str | int | float | bool | None],
    *,
    lock_already_held: bool = False,
    path: Path | str | None = None,
) -> None:
    target = _runtime_env_target_path(path or ENV_PATH)
    lock_context = (
        nullcontext()
        if lock_already_held
        else runtime_env_file_lock(path=target)
    )
    with lock_context:
        _identity, content = _read_runtime_env_snapshot(target)
        _cleanup_runtime_env_temporary_files(target)
        publish_bytes = _runtime_env_bytes_with_updates(content, updates)
        if publish_bytes != content:
            _atomic_replace_runtime_env_bytes(publish_bytes, path=target)


def _apply_runtime_env(updates: dict[str, str | int | float | bool | None]) -> None:
    forbidden_updates = sorted(
        {key.upper() for key in updates} & PROCESS_ONLY_ENV_KEYS
    )
    if forbidden_updates:
        raise ValueError(
            "Process-only environment keys cannot be applied from managed "
            "runtime settings: "
            + ", ".join(forbidden_updates)
        )
    for key, value in updates.items():
        env_key = key.upper()
        if value is None:
            os.environ.pop(env_key, None)
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES[env_key] = None
        elif isinstance(value, bool):
            applied_value = "true" if value else "false"
            os.environ[env_key] = applied_value
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES[env_key] = applied_value
        else:
            applied_value = str(value)
            os.environ[env_key] = applied_value
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES[env_key] = applied_value


def _deserialize_runtime_env_value(value: str) -> str:
    text = str(value)
    if (
        len(text) < 2
        or text[0] not in {'"', "'"}
        or text[-1] != text[0]
    ):
        return text
    body = text[1:-1]
    if text[0] == "'":
        return body
    decoded: list[str] = []
    escaped = False
    for character in body:
        if escaped:
            decoded.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        else:
            decoded.append(character)
    if escaped:
        decoded.append("\\")
    return "".join(decoded)


def _apply_runtime_env_file_to_process_environment(
    *,
    allowed_setting_keys: set[str] | frozenset[str] | None = None,
) -> None:
    entries = _env_entries(ENV_PATH)
    # Only settings governed by the runtime-settings lifecycle may be
    # reverse-applied to the current process.  Deployment/bootstrap settings
    # such as DATA_ROOT, DATABASE_URL and REDIS_URL may be present in the
    # managed file for service startup, but an in-process refresh must never
    # overwrite an explicit process/test value for those fields.
    selected_settings = RUNTIME_ENV_SETTINGS if allowed_setting_keys is None else allowed_setting_keys
    controlled_keys = {name.upper() for name in selected_settings}
    if allowed_setting_keys is None:
        controlled_keys |= set(DEPRECATED_ENV_KEYS)
    controlled_keys -= PROCESS_ONLY_ENV_KEYS
    for key in PROCESS_ONLY_ENV_KEYS:
        _RUNTIME_ENV_PROCESS_APPLIED_VALUES.pop(key, None)
    for key in controlled_keys:
        current_value = os.environ.get(key)
        if (
            key in _RUNTIME_ENV_PROCESS_APPLIED_VALUES
            and current_value != _RUNTIME_ENV_PROCESS_APPLIED_VALUES[key]
        ):
            # A caller/test changed this process value after the last managed
            # refresh.  Preserve that explicit override; other API/worker
            # processes still consume the shared file at their own refresh
            # boundary.
            continue
        if key in entries:
            applied_value = _deserialize_runtime_env_value(entries[key])
            os.environ[key] = applied_value
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES[key] = applied_value
        else:
            os.environ.pop(key, None)
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES[key] = None


def _redis_client():
    import redis

    settings = get_settings()
    return redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def _normalize_bridge_base_url(value: str | None) -> str:
    return (value or "").strip().rstrip("/")


def _normalize_bridge_resolve_ip(value: str | None) -> str:
    text = (value or "").strip()
    if not text or text == "__none__":
        return ""
    return text


def _hash_bridge_target(value: str | None) -> str:
    normalized = _canonical_bridge_target_url(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_hash_bridge_target(value: str | None) -> str:
    try:
        return _hash_bridge_target(value)
    except ValueError:
        return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _canonical_bridge_target_url(value: str | None) -> str:
    """Return a strict, DNS-free identity for one bridge provider URL.

    Empty remains empty so status payloads can report a missing target.  Any
    non-empty value must be an unambiguous absolute HTTP(S) URL.  Unicode host
    input is rejected rather than silently applying a platform-dependent IDNA
    mapping; explicit, round-trippable ASCII A-labels remain valid.
    """

    raw = str(value or "")
    if not raw:
        return ""
    if raw != raw.strip() or "\\" in raw:
        raise ValueError("bridge target URL contains ambiguous whitespace or separators")
    if any(char.isspace() or unicodedata.category(char).startswith("C") for char in raw):
        raise ValueError("bridge target URL contains whitespace or control characters")
    try:
        parsed = urlparse(raw)
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError("bridge target URL has malformed authority syntax") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("bridge target must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("bridge target URL must not contain userinfo")
    if not hostname:
        raise ValueError("bridge target URL is missing a hostname")
    if hostname.endswith(".."):
        raise ValueError("bridge target hostname has an ambiguous DNS root")
    hostname = hostname[:-1] if hostname.endswith(".") else hostname
    if not hostname or "%" in hostname or not hostname.isascii():
        raise ValueError("bridge target hostname must use unambiguous ASCII/IDNA form")
    try:
        ip_value = ipaddress.ip_address(hostname)
    except ValueError:
        labels = hostname.split(".")
        lower_hostname = hostname.lower()

        def numeric_address_token(label: str) -> bool:
            if label.isdigit():
                return True
            if label.startswith("0x") and len(label) > 2:
                return all(char in "0123456789abcdef" for char in label[2:])
            return False

        # Several URL/network stacks accept historical IPv4 spellings such as
        # 127.1, 2130706433 or 0x7f000001 even though ``ipaddress`` correctly
        # rejects them as non-canonical.  Never let those values fall through
        # to the DNS-name branch.  A legitimate alphanumeric provider label
        # (for example model42.example) is deliberately unaffected.
        if lower_hostname.isdigit() or all(
            numeric_address_token(label) for label in lower_hostname.split(".")
        ):
            raise ValueError("bridge target hostname uses a non-canonical IP spelling")
        if len(hostname) > 253 or any(not label or len(label) > 63 for label in labels):
            raise ValueError("bridge target hostname is not canonical DNS syntax")
        normalized_labels: list[str] = []
        for label in labels:
            lower_label = label.lower()
            if (
                not lower_label[0].isalnum()
                or not lower_label[-1].isalnum()
                or any(not (char.isalnum() or char == "-") for char in lower_label)
            ):
                raise ValueError("bridge target hostname is not canonical DNS syntax")
            if lower_label.startswith("xn--"):
                try:
                    unicode_label = lower_label.encode("ascii").decode("idna")
                    round_trip = unicode_label.encode("idna").decode("ascii").lower()
                except UnicodeError as exc:
                    raise ValueError("bridge target hostname has invalid IDNA syntax") from exc
                if round_trip != lower_label:
                    raise ValueError("bridge target hostname has ambiguous IDNA syntax")
            normalized_labels.append(lower_label)
        canonical_host = ".".join(normalized_labels)
        host_for_netloc = canonical_host
    else:
        if ip_value.is_unspecified:
            raise ValueError("bridge target hostname must not be an unspecified address")
        if (
            isinstance(ip_value, ipaddress.IPv6Address)
            and ip_value.ipv4_mapped is not None
        ):
            raise ValueError(
                "bridge target hostname must not use an IPv4-mapped IPv6 alias"
            )
        canonical_host = ip_value.compressed.lower()
        host_for_netloc = (
            f"[{canonical_host}]" if ip_value.version == 6 else canonical_host
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("bridge target URL has an invalid port") from exc
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    if not 1 <= int(port) <= 65535:
        raise ValueError("bridge target URL port is outside 1..65535")
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=f"{host_for_netloc}:{int(port)}",
    ).geturl().rstrip("/")


def _is_public_unicast_ip(value: str) -> bool:
    try:
        ip_value = ipaddress.ip_address(value)
    except ValueError:
        return False
    if (
        isinstance(ip_value, ipaddress.IPv6Address)
        and ip_value.ipv4_mapped is not None
    ):
        return False
    return bool(
        ip_value.is_global
        and not ip_value.is_private
        and not ip_value.is_loopback
        and not ip_value.is_link_local
        and not ip_value.is_multicast
        and not ip_value.is_reserved
        and not ip_value.is_unspecified
        and not getattr(ip_value, "is_site_local", False)
    )


def _validated_bridge_upstream_url(value: str | None) -> str:
    normalized = _canonical_bridge_target_url(value)
    if not normalized:
        raise ValueError("bridge upstream provider URL is required")
    parsed = urlparse(normalized)
    if parsed.scheme != "https":
        raise ValueError("bridge upstream provider URL must use HTTPS")
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError(
            "bridge upstream provider URL must not contain params, query, or fragment"
        )
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("bridge upstream provider URL cannot target localhost")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not _is_public_unicast_ip(hostname):
            raise ValueError(
                "bridge upstream provider URL must not target a non-public IP"
            )
    return normalized


def _validated_bridge_resolve_ip(value: str | None) -> str:
    normalized = _normalize_bridge_resolve_ip(value)
    if not normalized:
        return ""
    if not _is_public_unicast_ip(normalized):
        raise ValueError("bridge resolve IP must be one public unicast address")
    return str(ipaddress.ip_address(normalized))


def _validated_provider_endpoint_base_url(
    value: str | None,
    *,
    protocol: str,
    purpose: str,
) -> str:
    """Validate one provider base before any runtime or candidate side effect."""

    if protocol not in MODEL_API_PROTOCOL_ALLOWLIST:
        raise ValueError("provider API protocol must be openai or anthropic")
    if purpose not in {"chat", "graph", "embedding"}:
        raise ValueError("provider endpoint purpose is unsupported")
    if purpose == "embedding" and protocol != "openai":
        raise ValueError("embedding transport must remain OpenAI-compatible")
    normalized = _validated_bridge_upstream_url(value)
    path = (urlparse(normalized).path or "").rstrip("/").casefold()
    fixed_route = (
        "/embeddings"
        if purpose == "embedding"
        else ("/v1/messages" if protocol == "anthropic" else "/chat/completions")
    )
    if path.endswith(fixed_route):
        raise ValueError("provider base URL must not include the fixed request route")
    if protocol == "anthropic" and path.endswith("/v1"):
        raise ValueError(
            "Anthropic base URL must not end in /v1 because the client appends /v1/messages"
        )
    return normalized


def _validated_provider_target(
    *,
    base_url: str | None,
    resolve_ip: str | None,
    protocol: str,
    purpose: str,
) -> dict[str, str]:
    return {
        "protocol": protocol,
        "base_url": _validated_provider_endpoint_base_url(
            base_url,
            protocol=protocol,
            purpose=purpose,
        ),
        "resolve_ip": _validated_bridge_resolve_ip(resolve_ip),
    }


def _validated_desired_bridge_config(
    desired: Mapping[str, str | int],
) -> dict[str, str | int]:
    chat_api_protocol = str(desired.get("chat_api_protocol") or "")
    if chat_api_protocol not in MODEL_API_PROTOCOL_ALLOWLIST:
        raise ValueError("bridge chat API protocol must be openai or anthropic")
    embedding_api_protocol = str(desired.get("embedding_api_protocol") or "")
    if embedding_api_protocol not in EMBEDDING_API_PROTOCOL_ALLOWLIST:
        raise ValueError("bridge embedding API protocol must be openai")
    chat_target = _validated_provider_target(
        base_url=str(desired.get("chat_target_base_url") or ""),
        resolve_ip=str(desired.get("chat_resolve_ip") or ""),
        protocol=chat_api_protocol,
        purpose="chat",
    )
    embedding_target = _validated_provider_target(
        base_url=str(desired.get("embedding_target_base_url") or ""),
        resolve_ip=str(desired.get("embedding_resolve_ip") or ""),
        protocol=embedding_api_protocol,
        purpose="embedding",
    )
    return {
        "chat_api_protocol": chat_api_protocol,
        "embedding_api_protocol": embedding_api_protocol,
        "chat_target_base_url": chat_target["base_url"],
        "chat_resolve_ip": chat_target["resolve_ip"],
        "embedding_target_base_url": embedding_target["base_url"],
        "embedding_resolve_ip": embedding_target["resolve_ip"],
        "timeout": int(desired.get("timeout") or 0),
    }


def _validated_bridge_admin_token(value: str | None) -> str:
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
        or validation_value.casefold() in KNOWN_UNSAFE_BRIDGE_ADMIN_TOKENS
    ):
        raise RuntimeError(
            "MODEL_BRIDGE_ADMIN_TOKEN must be explicit and non-default when the model bridge is enabled"
        )
    return token


def _bridge_admin_headers(settings=None) -> dict[str, str]:
    settings = settings or get_settings()
    token = _validated_bridge_admin_token(settings.model_bridge_admin_token)
    return {"X-Bridge-Admin-Token": token}


def _bridge_base_url(settings=None) -> str:
    settings = settings or get_settings()
    return _normalize_bridge_base_url(
        settings.chat_base_url
        or model_bridge_client_base_url(settings.model_bridge_port)
    )


def _validated_bridge_admin_base_url(settings=None) -> str:
    settings = settings or get_settings()
    normalized = _canonical_bridge_target_url(_bridge_base_url(settings))
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").casefold()
    expected_host = (
        urlparse(model_bridge_client_base_url(settings.model_bridge_port)).hostname
        or ""
    ).casefold()
    if (
        parsed.scheme != "http"
        or host
        not in {
            "host.docker.internal",
            "localhost",
            "127.0.0.1",
            "::1",
            expected_host,
        }
        or (parsed.port or 80) != int(settings.model_bridge_port)
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Model bridge admin origin is not the exact local bridge endpoint")
    return normalized.rstrip("/")


def _bridge_http_request(
    method: str,
    path: str,
    *,
    settings=None,
    headers: Mapping[str, str] | None = None,
    json_payload: Mapping[str, Any] | None = None,
    timeout: float,
):
    if path not in {"/health", "/admin/config", "/admin/reload"}:
        raise ValueError("Model bridge request path is not allowlisted")
    base_url = _validated_bridge_admin_base_url(settings)
    with httpx.Client(trust_env=False, follow_redirects=False) as client:
        return client.request(
            method,
            f"{base_url}{path}",
            headers=dict(headers or {}),
            json=dict(json_payload) if json_payload is not None else None,
            timeout=timeout,
        )


def _bridge_target_is_self(value: str | None, settings=None) -> bool:
    try:
        normalized = _canonical_bridge_target_url(value)
    except ValueError:
        return False
    if not normalized:
        return False
    settings = settings or get_settings()
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower().rstrip(".")
    expected_host = (
        urlparse(model_bridge_client_base_url(settings.model_bridge_port)).hostname
        or ""
    ).lower().rstrip(".")
    self_host = host in {"host.docker.internal", "localhost", expected_host}
    try:
        ip_value = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        mapped = (
            ip_value.ipv4_mapped
            if isinstance(ip_value, ipaddress.IPv6Address)
            else None
        )
        self_host = bool(
            ip_value.is_loopback
            or ip_value.is_unspecified
            or (mapped is not None and (mapped.is_loopback or mapped.is_unspecified))
        )
    if not self_host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return port == settings.model_bridge_port


def _bridge_self_target_keys(desired: dict[str, str | int], settings=None) -> list[str]:
    settings = settings or get_settings()
    blocked: list[str] = []
    if _bridge_target_is_self(str(desired.get("chat_target_base_url") or ""), settings):
        blocked.append("CHAT_BASE_URL")
    if _bridge_target_is_self(str(desired.get("embedding_target_base_url") or ""), settings):
        blocked.append("EMBEDDING_BASE_URL")
    return blocked


def _desired_bridge_config(settings=None, env_entries: dict[str, str] | None = None) -> dict[str, str | int]:
    settings = settings or get_settings()
    env_entries = env_entries or _env_entries(ENV_PATH)
    return {
        "chat_api_protocol": str(
            env_entries.get("CHAT_API_PROTOCOL")
            or os.getenv("CHAT_API_PROTOCOL")
            or getattr(settings, "chat_api_protocol", "openai")
        ),
        "chat_target_base_url": _normalize_bridge_base_url(env_entries.get("CHAT_BASE_URL") or os.getenv("CHAT_BASE_URL") or ""),
        "chat_resolve_ip": _normalize_bridge_resolve_ip(env_entries.get("CHAT_RESOLVE_IP") or os.getenv("CHAT_RESOLVE_IP") or ""),
        "embedding_api_protocol": str(
            env_entries.get("EMBEDDING_API_PROTOCOL")
            or os.getenv("EMBEDDING_API_PROTOCOL")
            or getattr(settings, "embedding_api_protocol", "openai")
        ),
        "embedding_target_base_url": _normalize_bridge_base_url(env_entries.get("EMBEDDING_BASE_URL") or os.getenv("EMBEDDING_BASE_URL") or ""),
        "embedding_resolve_ip": _normalize_bridge_resolve_ip(env_entries.get("EMBEDDING_RESOLVE_IP") or os.getenv("EMBEDDING_RESOLVE_IP") or ""),
        "timeout": int(settings.model_request_timeout_seconds or 180),
    }


def model_bridge_status_payload(settings=None, env_entries: dict[str, str] | None = None) -> dict:
    settings = settings or get_settings()
    env_entries = env_entries or _env_entries(ENV_PATH)
    enabled = bool(settings.model_bridge_enabled)
    raw_desired = _desired_bridge_config(settings, env_entries)
    desired = raw_desired
    invalid_upstream = False
    if enabled:
        try:
            desired = _validated_desired_bridge_config(raw_desired)
            _validated_bridge_admin_token(settings.model_bridge_admin_token)
        except (ValueError, RuntimeError):
            invalid_upstream = True
    self_target_keys = _bridge_self_target_keys(desired, settings)
    payload: dict = {
        "enabled": enabled,
        "base_url": _bridge_base_url(settings) if enabled else "",
        "desired_chat_target_hash": _safe_hash_bridge_target(str(desired["chat_target_base_url"])),
        "desired_embedding_target_hash": _safe_hash_bridge_target(str(desired["embedding_target_base_url"])),
        "desired_chat_api_protocol": str(desired["chat_api_protocol"]),
        "desired_embedding_api_protocol": str(
            desired["embedding_api_protocol"]
        ),
        "reachable": None,
        "admin_available": None,
        "config_matches": None,
        "chat_target_is_bridge": "CHAT_BASE_URL" in self_target_keys,
        "embedding_target_is_bridge": "EMBEDDING_BASE_URL" in self_target_keys,
        "self_target_blocked": bool(self_target_keys),
        "warnings": [],
    }
    if not enabled:
        return payload
    if self_target_keys:
        payload["warnings"].append(
            "Model bridge target points to the bridge itself: " + ", ".join(self_target_keys) + ". Use the real provider base URL."
        )
    if not desired["chat_target_base_url"]:
        payload["warnings"].append("CHAT_BASE_URL is required when MODEL_BRIDGE_ENABLED=true.")
    if not desired["embedding_target_base_url"]:
        payload["warnings"].append("EMBEDDING_BASE_URL is required when MODEL_BRIDGE_ENABLED=true.")
    if invalid_upstream:
        payload["reachable"] = False
        payload["admin_available"] = False
        payload["config_matches"] = False
        payload["warnings"].append(
            "Model bridge provider configuration or admin credential failed validation."
        )
        return payload
    try:
        health = _bridge_http_request(
            "GET", "/health", settings=settings, timeout=3.0
        )
        payload["reachable"] = health.status_code == 200
        if health.status_code == 200:
            health_json = health.json()
            payload["config_version"] = health_json.get("config_version")
            payload["chat_target_hash"] = health_json.get("chat_target_hash")
            payload["embedding_target_hash"] = health_json.get("embedding_target_hash")
            payload["routes"] = health_json.get("routes") or {}
            payload["chat_api_protocol"] = health_json.get("chat_api_protocol")
            payload["embedding_api_protocol"] = health_json.get(
                "embedding_api_protocol"
            )
    except Exception:
        payload["reachable"] = False
        payload["warnings"].append(
            "Model bridge health check failed with a transport error."
        )
        return payload

    try:
        config_response = _bridge_http_request(
            "GET",
            "/admin/config",
            settings=settings,
            headers=_bridge_admin_headers(settings),
            timeout=3.0,
        )
        payload["admin_available"] = config_response.status_code == 200
        if config_response.status_code != 200:
            payload["warnings"].append(f"Model bridge admin config returned HTTP {config_response.status_code}.")
            return payload
        config = config_response.json()
        payload["config_version"] = config.get("config_version")
        payload["chat_target_hash"] = config.get("chat_target_hash")
        payload["embedding_target_hash"] = config.get("embedding_target_hash")
        payload["config_matches"] = (
            config.get("chat_api_protocol") == desired["chat_api_protocol"]
            and
            config.get("embedding_api_protocol")
            == desired["embedding_api_protocol"]
            and
            _canonical_bridge_target_url(config.get("chat_target_base_url"))
            == desired["chat_target_base_url"]
            and _canonical_bridge_target_url(
                config.get("embedding_target_base_url")
            )
            == desired["embedding_target_base_url"]
            and _normalize_bridge_resolve_ip(config.get("chat_resolve_ip")) == desired["chat_resolve_ip"]
            and _normalize_bridge_resolve_ip(config.get("embedding_resolve_ip")) == desired["embedding_resolve_ip"]
            and int(config.get("timeout") or 0) == int(desired["timeout"])
            and not self_target_keys
        )
        if not payload["config_matches"]:
            payload["warnings"].append("Model bridge config does not match current .env targets.")
    except Exception:
        payload["admin_available"] = False
        payload["warnings"].append(
            "Model bridge admin config failed with a transport error."
        )
    return payload


def reload_model_bridge(settings=None, env_entries: dict[str, str] | None = None) -> dict:
    settings = settings or get_settings()
    env_entries = env_entries or _env_entries(ENV_PATH)
    if not settings.model_bridge_enabled:
        return {"attempted": False, "reason": "model_bridge_disabled"}
    raw_desired = _desired_bridge_config(settings, env_entries)
    try:
        desired = _validated_desired_bridge_config(raw_desired)
        _validated_bridge_admin_token(settings.model_bridge_admin_token)
    except (ValueError, RuntimeError):
        return {
            "attempted": False,
            "ok": False,
            "error": "Model bridge provider configuration or admin credential failed validation.",
        }
    self_target_keys = _bridge_self_target_keys(desired, settings)
    if self_target_keys:
        return {
            "attempted": True,
            "ok": False,
            "self_target_blocked": True,
            "error": "Model bridge target points to the bridge itself: " + ", ".join(self_target_keys),
        }
    if not desired["chat_target_base_url"] or not desired["embedding_target_base_url"]:
        return {"attempted": False, "ok": False, "error": "CHAT_BASE_URL and EMBEDDING_BASE_URL are required when MODEL_BRIDGE_ENABLED=true."}
    try:
        response = _bridge_http_request(
            "POST",
            "/admin/reload",
            settings=settings,
            headers=_bridge_admin_headers(settings),
            json_payload=desired,
            timeout=5.0,
        )
        if response.status_code != 200:
            return {
                "attempted": True,
                "ok": False,
                "status_code": response.status_code,
                "error": sanitize_error_message(f"Model bridge admin reload failed with HTTP {response.status_code}"),
            }
        payload = response.json()
        return {
            "attempted": True,
            "ok": True,
            "config_version": payload.get("config_version"),
            "chat_target_hash": payload.get("chat_target_hash"),
            "embedding_target_hash": payload.get("embedding_target_hash"),
        }
    except Exception as exc:
        return {"attempted": True, "ok": False, "error": public_exception_message(exc)}


def preflight_model_bridge_reload(
    *,
    settings,
    env_entries: dict[str, str],
) -> dict[str, Any]:
    """Validate desired and current bridge identities without mutating either."""

    desired = _validated_desired_bridge_config(
        _desired_bridge_config(settings, env_entries)
    )
    _validated_bridge_admin_token(settings.model_bridge_admin_token)
    self_target_keys = _bridge_self_target_keys(desired, settings)
    if self_target_keys:
        raise ValueError("Model bridge prospective target points to the bridge itself")
    try:
        response = _bridge_http_request(
            "GET",
            "/admin/config",
            settings=settings,
            headers=_bridge_admin_headers(settings),
            timeout=3.0,
        )
        if response.status_code != 200:
            raise RuntimeError("model bridge admin preflight failed")
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("model bridge admin preflight was malformed")
        previous = _validated_desired_bridge_config(
            {
                "chat_api_protocol": payload.get("chat_api_protocol"),
                "embedding_api_protocol": payload.get("embedding_api_protocol"),
                "chat_target_base_url": payload.get("chat_target_base_url"),
                "chat_resolve_ip": payload.get("chat_resolve_ip"),
                "embedding_target_base_url": payload.get(
                    "embedding_target_base_url"
                ),
                "embedding_resolve_ip": payload.get("embedding_resolve_ip"),
                "timeout": payload.get("timeout"),
            }
        )
    except Exception:
        raise ValueError(
            "Model bridge admin preflight failed before runtime mutation"
        ) from None
    return {"desired": desired, "previous": previous}


def sync_model_bridge_runtime_config(settings=None, env_entries: dict[str, str] | None = None, *, raise_on_error: bool = True) -> dict:
    settings = settings or get_settings()
    env_entries = env_entries or _env_entries(ENV_PATH)
    if not settings.model_bridge_enabled:
        return {"attempted": False, "reason": "model_bridge_disabled"}
    result = reload_model_bridge(settings=settings, env_entries=env_entries)
    failed = result.get("ok") is False or (result.get("attempted") is False and result.get("reason") != "model_bridge_disabled")
    if failed and raise_on_error:
        raise RuntimeError(f"Model bridge runtime config sync failed: {result.get('error') or result}")
    return result


def current_runtime_settings_version() -> str | None:
    with suppress(Exception):
        version = _redis_client().get(SETTINGS_VERSION_REDIS_KEY)
        if version:
            return str(version)
    return _LAST_RUNTIME_SETTINGS_VERSION


def _version_hash(
    changed_keys: list[str],
    created_at: str,
    env_identity_hash: str | None,
) -> str:
    payload = {
        "protocol_version": "runtime_settings_version_v2",
        "changed_keys": sorted(set(changed_keys)),
        "created_at": created_at,
        "env_identity_hash": env_identity_hash,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _local_runtime_refresh(
    version: str | None = None,
    *,
    changed_keys: list[str] | None = None,
    apply_rebuild: bool = False,
) -> None:
    global _LAST_RUNTIME_SETTINGS_VERSION
    allowed = set(HOT_RELOAD_SETTINGS)
    if apply_rebuild:
        allowed.update(REBUILD_REQUIRED_SETTINGS)
    if changed_keys is not None:
        normalized = {str(key).lower() for key in changed_keys}
        allowed.intersection_update(normalized)
    _apply_runtime_env_file_to_process_environment(allowed_setting_keys=allowed)
    get_settings.cache_clear()
    get_settings()
    from app.services.cache_manager import clear_cache_manager

    clear_cache_manager()
    if version is not None:
        _LAST_RUNTIME_SETTINGS_VERSION = version


def publish_runtime_settings_version(
    changed_keys: list[str],
    source: str = "api",
    *,
    idempotency_key: str | None = None,
) -> dict:
    managed_env_identity = runtime_env_file_identity(ENV_PATH)
    managed_env_identity_hash = (
        str(managed_env_identity.get("identity_hash"))
        if managed_env_identity.get("exists")
        else None
    )
    normalized_changed_keys = sorted(set(changed_keys))
    stored_source = str(source or "api")
    if idempotency_key is not None:
        raw_key = str(idempotency_key).strip()
        if not raw_key:
            raise ValueError("runtime settings idempotency_key cannot be empty")
        candidate_source = f"{stored_source}:{raw_key}"
        stored_source = (
            candidate_source
            if len(candidate_source) <= 64
            else f"{stored_source[:31]}:{hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:32]}"
        )

    from app.db import SessionLocal
    from app.models import RuntimeSettingsVersion

    with SessionLocal() as session:
        if session.get_bind().dialect.name == "postgresql":
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
                {"scope": RUNTIME_SETTINGS_PUBLICATION_ADVISORY_SCOPE},
            )
        existing = None
        if idempotency_key is not None:
            existing = session.scalar(
                select(RuntimeSettingsVersion)
                .where(RuntimeSettingsVersion.source == stored_source)
                .order_by(RuntimeSettingsVersion.created_at.desc())
            )
        if existing is not None:
            if sorted(existing.changed_keys_json or []) != normalized_changed_keys:
                raise RuntimeError(
                    "Runtime settings idempotency key was reused with different changed keys"
                )
            if existing.managed_env_identity_hash != managed_env_identity_hash:
                raise RuntimeError(
                    "Runtime settings idempotency key was reused with a different managed env identity"
                )
            message = {
                "version_hash": existing.version_hash,
                "changed_keys": normalized_changed_keys,
                "source": stored_source,
                "created_at": existing.created_at.replace(
                    tzinfo=existing.created_at.tzinfo or timezone.utc
                ).isoformat(),
            }
        else:
            created_at_value = datetime.utcnow()
            created_at = created_at_value.replace(tzinfo=timezone.utc).isoformat()
            version_hash = _version_hash(
                normalized_changed_keys,
                created_at,
                managed_env_identity_hash,
            )
            message = {
                "version_hash": version_hash,
                "changed_keys": normalized_changed_keys,
                "source": stored_source,
                "created_at": created_at,
            }
            session.add(
                RuntimeSettingsVersion(
                    version_hash=version_hash,
                    changed_keys_json=normalized_changed_keys,
                    source=stored_source,
                    managed_env_identity_hash=managed_env_identity_hash,
                    created_at=created_at_value,
                )
            )
            session.commit()

    try:
        client = _redis_client()
        client.set(SETTINGS_VERSION_REDIS_KEY, message["version_hash"])
        client.publish(
            SETTINGS_VERSION_CHANNEL,
            json.dumps(message, ensure_ascii=False, sort_keys=True),
        )
    except Exception as exc:
        # The committed RuntimeSettingsVersion row is the durable replay card.
        # Keep env/bridge/current-process state on the same new identity and
        # expose a typed pending broadcast instead of rolling active bytes back
        # behind an already-committed audit fact.
        local_refresh_error_type: str | None = None
        try:
            _local_runtime_refresh(
                message["version_hash"],
                changed_keys=normalized_changed_keys,
                apply_rebuild=stored_source.startswith("runtime_settings_"),
            )
        except Exception as local_exc:
            local_refresh_error_type = type(local_exc).__name__
        return {
            **message,
            "runtime_version_broadcast": False,
            "broadcast_pending": True,
            "broadcast_error_type": type(exc).__name__,
            "local_refresh_pending": local_refresh_error_type is not None,
            "local_refresh_error_type": local_refresh_error_type,
        }
    local_refresh_error_type = None
    try:
        _local_runtime_refresh(
            message["version_hash"],
            changed_keys=normalized_changed_keys,
            apply_rebuild=stored_source.startswith("runtime_settings_"),
        )
    except Exception as exc:
        local_refresh_error_type = type(exc).__name__
    return {
        **message,
        "runtime_version_broadcast": True,
        "broadcast_pending": False,
        "broadcast_error_type": None,
        "local_refresh_pending": local_refresh_error_type is not None,
        "local_refresh_error_type": local_refresh_error_type,
    }


def reconcile_runtime_settings_version_broadcast() -> dict[str, Any]:
    """Replay the latest durable version row only when active bytes still match."""

    from app.db import SessionLocal
    from app.models import RuntimeSettingsVersion

    with SessionLocal() as session:
        latest = session.scalar(
            select(RuntimeSettingsVersion).order_by(
                RuntimeSettingsVersion.created_at.desc(),
                RuntimeSettingsVersion.id.desc(),
            )
        )
        if latest is None:
            return {"replayed": False, "reason": "no_durable_version"}
        current_identity = runtime_env_file_identity(ENV_PATH)
        current_identity_hash = (
            str(current_identity.get("identity_hash"))
            if current_identity.get("exists")
            else None
        )
        if latest.managed_env_identity_hash != current_identity_hash:
            return {
                "replayed": False,
                "reason": "managed_env_identity_mismatch",
                "version_hash": latest.version_hash,
            }
        message = {
            "version_hash": latest.version_hash,
            "changed_keys": sorted(latest.changed_keys_json or []),
            "source": latest.source,
            "created_at": latest.created_at.replace(
                tzinfo=latest.created_at.tzinfo or timezone.utc
            ).isoformat(),
        }
    client = _redis_client()
    client.set(SETTINGS_VERSION_REDIS_KEY, message["version_hash"])
    client.publish(
        SETTINGS_VERSION_CHANNEL,
        json.dumps(message, ensure_ascii=False, sort_keys=True),
    )
    return {
        "replayed": True,
        "reason": "durable_version_replayed",
        "version_hash": message["version_hash"],
    }


def refresh_runtime_settings_if_needed(
    force: bool = False,
    *,
    sync_bridge: bool = True,
) -> dict:
    global _LAST_RUNTIME_SETTINGS_VERSION
    version: str | None = None
    redis_error: str | None = None
    bridge_sync: dict | None = None
    try:
        version = _redis_client().get(SETTINGS_VERSION_REDIS_KEY)
    except Exception as exc:  # pragma: no cover - exercised by integration/runtime checks
        redis_error = str(exc)

    should_refresh = force or (version is not None and version != _LAST_RUNTIME_SETTINGS_VERSION)
    if should_refresh:
        changed_keys: list[str] | None = None
        apply_rebuild = False
        if version is not None:
            with suppress(Exception):
                from app.db import SessionLocal
                from app.models import RuntimeSettingsVersion

                with SessionLocal() as session:
                    row = session.scalar(
                        select(RuntimeSettingsVersion).where(
                            RuntimeSettingsVersion.version_hash == version
                        )
                    )
                    if row is not None:
                        changed_keys = list(row.changed_keys_json or [])
                        apply_rebuild = str(row.source or "").startswith(
                            "runtime_settings_"
                        )
        _local_runtime_refresh(
            version,
            changed_keys=changed_keys,
            apply_rebuild=apply_rebuild,
        )
        if sync_bridge:
            bridge_sync = sync_model_bridge_runtime_config(settings=get_settings())
    elif version is not None:
        _LAST_RUNTIME_SETTINGS_VERSION = version
    result = {
        "refreshed": should_refresh,
        "runtime_settings_version": version or _LAST_RUNTIME_SETTINGS_VERSION,
        "redis_error": redis_error,
    }
    if bridge_sync is not None:
        result["model_bridge_sync"] = bridge_sync
    return result


def _env_keys(path: Path) -> tuple[set[str], list[str]]:
    keys: set[str] = set()
    bom_keys: list[str] = []
    if not path.exists():
        return keys, bom_keys
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue
        raw_key = raw_line.split("=", 1)[0].strip()
        clean_key = raw_key.lstrip("\ufeff").upper()
        if raw_key.startswith("\ufeff"):
            bom_keys.append(clean_key)
        keys.add(clean_key)
    return keys, bom_keys


def _env_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.exists():
        return entries
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue
        raw_key, value = raw_line.split("=", 1)
        key = raw_key.strip().lstrip("\ufeff").upper()
        entries.setdefault(key, value)
    return entries


def _settings_from_root_env(base: Settings | None = None) -> Settings:
    """Validate the sole root env as the value shown by the Settings API.

    The current process model remains the lifecycle baseline: hot values may
    already match the file, rebuild values may be waiting for promotion and
    service values may be waiting for recreation.  No second value snapshot is
    read from PostgreSQL or another file.
    """

    model = (base or get_settings()).model_dump(mode="python")
    entries = _env_entries(ENV_PATH)
    for setting_key in sorted(RUNTIME_ENV_SETTINGS):
        env_key = setting_key.upper()
        if env_key in entries:
            model[setting_key] = _deserialize_runtime_env_value(entries[env_key])
    return Settings.model_validate(model)


def _normalized_runtime_env_bytes(
    content: bytes,
    *,
    remove_deprecated: bool,
) -> bytes:
    text = content.decode("utf-8")
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        clean_line = line.lstrip("\ufeff")
        stripped = clean_line.strip()
        if (
            stripped
            and not stripped.startswith("#")
            and "=" in clean_line
        ):
            key = clean_line.split("=", 1)[0].strip().upper()
            if remove_deprecated and key in DEPRECATED_ENV_KEYS:
                continue
        cleaned_lines.append(clean_line)
    return ("\n".join(cleaned_lines).rstrip() + "\n").encode("utf-8")


def _rollback_runtime_env_recovery_state(
    state: Mapping[str, Any],
    *,
    outcome: str,
    error_type: str | None,
) -> dict[str, Any]:
    before_bytes = _read_runtime_env_recovery_backup(state)
    position, _identity = _runtime_env_recovery_target_position(state)
    if position == "published":
        if state.get("before_exists"):
            _atomic_replace_runtime_env_bytes(
                before_bytes,
                path=str(state["target_path"]),
                fsync_callable=_RUNTIME_ENV_RECOVERY_FSYNC,
                replace_callable=_RUNTIME_ENV_RECOVERY_REPLACE,
            )
        else:
            _durable_unlink_runtime_env_path(
                _runtime_env_target_path(str(state["target_path"]))
            )
    elif position != "before":
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env target no longer matches either side of the durable "
            f"receipt; recovery_id={state['recovery_id']}"
        )
    restored_position, restored_identity = (
        _runtime_env_recovery_target_position(state)
    )
    if restored_position != "before":
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env durable rollback could not verify the confirmed bytes; "
            f"recovery_id={state['recovery_id']}"
        )
    resolved = {
        **dict(state),
        "phase": "rolled_back",
        "after_identity_hash": restored_identity.get("identity_hash"),
        "resolution": outcome,
    }
    _write_runtime_env_recovery_state(resolved)
    fact = _record_runtime_env_recovery_audit(
        resolved,
        outcome=outcome,
        error_type=error_type,
    )
    _cleanup_runtime_env_recovery_files(resolved)
    return fact


def _rollback_after_runtime_env_publish_failure(
    state: Mapping[str, Any],
    publish_error: BaseException,
) -> RuntimeEnvFilePublishError:
    try:
        fact = _rollback_runtime_env_recovery_state(
            state,
            outcome="durable_rollback_after_publish_failure",
            error_type=type(publish_error).__name__,
        )
    except BaseException as rollback_error:
        raise RuntimeEnvFileRecoveryRequired(
            "Runtime env publish and durable rollback both failed; "
            f"recovery_id={state['recovery_id']}; "
            f"publish_error_type={type(publish_error).__name__}; "
            f"rollback_error_type={type(rollback_error).__name__}"
        ) from rollback_error
    return RuntimeEnvFilePublishError(
        "Runtime env publication failed and confirmed bytes were restored "
        f"durably; recovery_id={state['recovery_id']}; "
        f"publish_error={publish_error}",
        recovery_fact=fact,
    )


def _publish_runtime_env_bytes_auto_commit(
    *,
    before_identity: Mapping[str, Any],
    before_bytes: bytes | None,
    publish_bytes: bytes,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Publish a standalone writer update with a crash-resolvable receipt."""

    target = _runtime_env_target_path(path)
    state = _prepare_runtime_env_recovery(
        before_identity=before_identity,
        before_bytes=before_bytes,
        publish_bytes=publish_bytes,
        commit_policy="publish_is_commit",
        path=target,
    )
    try:
        _atomic_replace_runtime_env_bytes(publish_bytes, path=target)
        after_identity = runtime_env_file_identity(target)
        if (
            after_identity.get("file_sha256")
            != state["publish_file_sha256"]
            or after_identity.get("size_bytes")
            != state["publish_size_bytes"]
        ):
            raise RuntimeEnvFileConflict(
                "Runtime env auto-publication did not produce prepared bytes"
            )
        state = {
            **state,
            "phase": "published_pending_commit",
            "after_identity_hash": after_identity["identity_hash"],
        }
        _write_runtime_env_recovery_state(state)
    except BaseException as publish_error:
        raise _rollback_after_runtime_env_publish_failure(
            state,
            publish_error,
        ) from publish_error

    committed = {
        **state,
        "phase": "committed",
        "resolution": "standalone_publish_commit_point",
    }
    try:
        _write_runtime_env_recovery_state(committed)
    except BaseException as commit_error:
        raise _rollback_after_runtime_env_publish_failure(
            state,
            commit_error,
        ) from commit_error
    try:
        _cleanup_runtime_env_recovery_files(committed)
    except BaseException as cleanup_error:
        fact = _runtime_env_recovery_fact(
            committed,
            outcome="publish_committed_recovery_cleanup_pending",
        )
        raise RuntimeEnvFilePublishError(
            "Runtime env publish crossed its durable commit point but recovery "
            f"artifact cleanup failed; recovery_id={state['recovery_id']}; "
            f"cleanup_error={cleanup_error}",
            recovery_fact=fact,
        ) from cleanup_error
    return after_identity


def prepare_runtime_env_file_updates_exact(
    updates: Mapping[str, str | int | float | bool | None],
    *,
    expected_identity: Mapping[str, Any],
    lock_already_held: bool = False,
) -> RuntimeEnvFileMutationReceipt:
    """Publish exact managed-env bytes but keep a caller-commit recovery receipt.

    The caller must durably commit its PostgreSQL intent before invoking
    :func:`commit_runtime_env_file_mutation`.  A crash in between leaves the
    versioned recovery journal and blocks every other writer; raw env bytes
    never enter the durable intent or public receipt.
    """

    if not isinstance(updates, Mapping) or not updates:
        raise ValueError("Runtime env exact updates must be a non-empty mapping")
    normalized_updates = {str(key).upper(): value for key, value in updates.items()}
    if any(not key or key != key.strip() for key in normalized_updates):
        raise ValueError("Runtime env exact update keys must be canonical")
    lock_context = (
        nullcontext()
        if lock_already_held
        else runtime_env_file_lock(path=ENV_PATH)
    )
    with lock_context:
        _require_runtime_env_recovery_resolved(ENV_PATH)
        before_identity, before_bytes = _require_runtime_env_identity(
            expected_identity,
            path=ENV_PATH,
        )
        if before_bytes is None:
            raise RuntimeEnvFileConflict(
                "Exact runtime env update requires an existing managed file"
            )
        _cleanup_runtime_env_temporary_files(ENV_PATH)
        publish_bytes = _runtime_env_bytes_with_updates(
            before_bytes,
            normalized_updates,
        )
        if publish_bytes == before_bytes:
            return RuntimeEnvFileMutationReceipt(
                protocol_version=RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION,
                target_path=str(_runtime_env_target_path(ENV_PATH)),
                changed=False,
                before_identity=before_identity,
                after_identity=before_identity,
                before_bytes=before_bytes,
                recovery_id=None,
                recovery_journal_path_hash=None,
            )
        recovery_state = _prepare_runtime_env_recovery(
            before_identity=before_identity,
            before_bytes=before_bytes,
            publish_bytes=publish_bytes,
            commit_policy="caller_commit_required",
            path=ENV_PATH,
        )
        try:
            _atomic_replace_runtime_env_bytes(publish_bytes, path=ENV_PATH)
            after_identity = runtime_env_file_identity(ENV_PATH)
            if (
                after_identity.get("file_sha256")
                != recovery_state["publish_file_sha256"]
                or after_identity.get("size_bytes")
                != recovery_state["publish_size_bytes"]
            ):
                raise RuntimeEnvFileConflict(
                    "Runtime env exact update did not produce the prepared bytes"
                )
            recovery_state = {
                **recovery_state,
                "phase": "published_pending_commit",
                "after_identity_hash": after_identity["identity_hash"],
            }
            _write_runtime_env_recovery_state(recovery_state)
        except BaseException as publish_error:
            raise _rollback_after_runtime_env_publish_failure(
                recovery_state,
                publish_error,
            ) from publish_error
        journal_path, _backup_path, _audit_path = _runtime_env_recovery_paths(
            ENV_PATH
        )
        return RuntimeEnvFileMutationReceipt(
            protocol_version=RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION,
            target_path=str(_runtime_env_target_path(ENV_PATH)),
            changed=True,
            before_identity=before_identity,
            after_identity=after_identity,
            before_bytes=before_bytes,
            recovery_id=recovery_state["recovery_id"],
            recovery_journal_path_hash=_runtime_env_recovery_path_hash(
                journal_path
            ),
        )


def normalize_env_file_exact(
    *,
    expected_identity: Mapping[str, Any],
    remove_deprecated: bool = True,
    lock_already_held: bool = False,
) -> RuntimeEnvFileMutationReceipt:
    """CAS-normalize exactly the confirmed env bytes under the writer fence."""

    lock_context = (
        nullcontext()
        if lock_already_held
        else runtime_env_file_lock(path=ENV_PATH)
    )
    with lock_context:
        _require_runtime_env_recovery_resolved(ENV_PATH)
        before_identity, before_bytes = _require_runtime_env_identity(
            expected_identity,
            path=ENV_PATH,
        )
        _cleanup_runtime_env_temporary_files(ENV_PATH)
        if before_bytes is None:
            return RuntimeEnvFileMutationReceipt(
                protocol_version=RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION,
                target_path=str(_runtime_env_target_path(ENV_PATH)),
                changed=False,
                before_identity=before_identity,
                after_identity=before_identity,
                before_bytes=None,
                recovery_id=None,
                recovery_journal_path_hash=None,
            )
        normalized = _normalized_runtime_env_bytes(
            before_bytes,
            remove_deprecated=remove_deprecated,
        )
        if normalized == before_bytes:
            return RuntimeEnvFileMutationReceipt(
                protocol_version=RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION,
                target_path=str(_runtime_env_target_path(ENV_PATH)),
                changed=False,
                before_identity=before_identity,
                after_identity=before_identity,
                before_bytes=before_bytes,
                recovery_id=None,
                recovery_journal_path_hash=None,
            )
        recovery_state = _prepare_runtime_env_recovery(
            before_identity=before_identity,
            before_bytes=before_bytes,
            publish_bytes=normalized,
            commit_policy="caller_commit_required",
            path=ENV_PATH,
        )
        try:
            _atomic_replace_runtime_env_bytes(normalized, path=ENV_PATH)
            after_identity = runtime_env_file_identity(ENV_PATH)
            if (
                after_identity.get("file_sha256")
                != recovery_state["publish_file_sha256"]
                or after_identity.get("size_bytes")
                != recovery_state["publish_size_bytes"]
            ):
                raise RuntimeEnvFileConflict(
                    "Runtime env publication did not produce the prepared bytes"
                )
            recovery_state = {
                **recovery_state,
                "phase": "published_pending_commit",
                "after_identity_hash": after_identity["identity_hash"],
            }
            _write_runtime_env_recovery_state(recovery_state)
        except BaseException as publish_error:
            raise _rollback_after_runtime_env_publish_failure(
                recovery_state,
                publish_error,
            ) from publish_error
        journal_path, _backup_path, _audit_path = (
            _runtime_env_recovery_paths(ENV_PATH)
        )
        return RuntimeEnvFileMutationReceipt(
            protocol_version=RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION,
            target_path=str(_runtime_env_target_path(ENV_PATH)),
            changed=True,
            before_identity=before_identity,
            after_identity=after_identity,
            before_bytes=before_bytes,
            recovery_id=recovery_state["recovery_id"],
            recovery_journal_path_hash=(
                _runtime_env_recovery_path_hash(journal_path)
            ),
        )


def restore_runtime_env_file_mutation(
    receipt: RuntimeEnvFileMutationReceipt,
    *,
    lock_already_held: bool = False,
) -> bool:
    """Restore exact pre-mutation bytes after a downstream rollback."""

    if (
        not isinstance(receipt, RuntimeEnvFileMutationReceipt)
        or receipt.protocol_version != RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION
        or receipt.target_path != str(_runtime_env_target_path(ENV_PATH))
    ):
        raise RuntimeEnvFileConflict(
            "Runtime env rollback receipt is invalid"
        )
    if not receipt.changed:
        return False
    if not receipt.recovery_id or not receipt.recovery_journal_path_hash:
        raise RuntimeEnvFileRecoveryRequired(
            "Changed runtime env receipt has no durable recovery identity"
        )
    lock_context = (
        nullcontext()
        if lock_already_held
        else runtime_env_file_lock(path=ENV_PATH)
    )
    with lock_context:
        state = _load_runtime_env_recovery_state(ENV_PATH)
        if (
            state is None
            or state["recovery_id"] != receipt.recovery_id
            or _runtime_env_recovery_path_hash(
                _runtime_env_recovery_paths(ENV_PATH)[0]
            )
            != receipt.recovery_journal_path_hash
            or state["phase"] not in {
                "prepared",
                "published_pending_commit",
            }
        ):
            raise RuntimeEnvFileRecoveryRequired(
                "Runtime env durable rollback receipt is missing or no longer pending"
            )
        if (
            receipt.before_bytes is None
            or hashlib.sha256(receipt.before_bytes).hexdigest()
            != state["before_file_sha256"]
        ):
            raise RuntimeEnvFileRecoveryRequired(
                "Runtime env in-memory receipt does not match its durable backup"
            )
        _rollback_runtime_env_recovery_state(
            state,
            outcome="durable_rollback_after_downstream_failure",
            error_type=None,
        )
        return True


def commit_runtime_env_file_mutation(
    receipt: RuntimeEnvFileMutationReceipt,
    *,
    lock_already_held: bool = False,
) -> bool:
    """Cross the env commit point after the caller's durable DB commit."""

    if (
        not isinstance(receipt, RuntimeEnvFileMutationReceipt)
        or receipt.protocol_version != RUNTIME_ENV_FILE_CAS_PROTOCOL_VERSION
        or receipt.target_path != str(_runtime_env_target_path(ENV_PATH))
    ):
        raise RuntimeEnvFileConflict("Runtime env commit receipt is invalid")
    if not receipt.changed:
        return False
    if not receipt.recovery_id or not receipt.recovery_journal_path_hash:
        raise RuntimeEnvFileRecoveryRequired(
            "Changed runtime env receipt has no durable commit identity"
        )
    lock_context = (
        nullcontext()
        if lock_already_held
        else runtime_env_file_lock(path=ENV_PATH)
    )
    with lock_context:
        state = _load_runtime_env_recovery_state(ENV_PATH)
        if (
            state is None
            or state["recovery_id"] != receipt.recovery_id
            or state["phase"] != "published_pending_commit"
            or _runtime_env_recovery_path_hash(
                _runtime_env_recovery_paths(ENV_PATH)[0]
            )
            != receipt.recovery_journal_path_hash
        ):
            raise RuntimeEnvFileRecoveryRequired(
                "Runtime env durable commit receipt is missing or not publish-pending"
            )
        position, current_identity = _runtime_env_recovery_target_position(state)
        if (
            position != "published"
            or current_identity != receipt.after_identity
        ):
            raise RuntimeEnvFileConflict(
                "Runtime env changed before its durable commit point"
            )
        committed = {
            **state,
            "phase": "committed",
            "resolution": "caller_database_commit_completed",
        }
        try:
            _write_runtime_env_recovery_state(committed)
        except BaseException as marker_error:
            pending_restore_error: BaseException | None = None
            pending_verified = False
            try:
                durable_state = _load_runtime_env_recovery_state(ENV_PATH)
                if durable_state != state:
                    _write_runtime_env_recovery_state(state)
                pending_verified = (
                    _load_runtime_env_recovery_state(ENV_PATH) == state
                )
            except BaseException as exc:
                pending_restore_error = exc
            fact = {
                **_runtime_env_recovery_fact(
                    state,
                    outcome=(
                        "caller_database_commit_completed_env_commit_marker_pending"
                    ),
                ),
                "phase": "published_pending_commit",
                "target_position": "published",
                "pending_receipt_verified": pending_verified,
                "commit_marker_error_type": type(marker_error).__name__,
                "pending_restore_error_type": (
                    type(pending_restore_error).__name__
                    if pending_restore_error is not None
                    else None
                ),
            }
            raise RuntimeEnvFileRecoveryRequired(
                "Caller database commit completed, but the runtime env commit "
                "marker was not made durable; explicit commit recovery is "
                "required",
                recovery_fact=fact,
                recovery_id=str(state["recovery_id"]),
            ) from marker_error
        try:
            _cleanup_runtime_env_recovery_files(committed)
        except BaseException as cleanup_error:
            fact = _runtime_env_recovery_fact(
                committed,
                outcome="caller_commit_completed_recovery_cleanup_pending",
            )
            raise RuntimeEnvFilePublishError(
                "Runtime env and caller database commit points are durable but "
                "recovery artifact cleanup failed; "
                f"recovery_id={state['recovery_id']}; "
                f"cleanup_error_type={type(cleanup_error).__name__}",
                recovery_fact=fact,
            ) from cleanup_error
        return True


def resolve_runtime_env_file_recovery(
    *,
    recovery_id: str,
    action: str,
    expected_current_identity: Mapping[str, Any],
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Explicitly resolve a crash receipt after exact identity confirmation."""

    target = _runtime_env_target_path(path)
    if action not in {"commit", "rollback"}:
        raise ValueError("Runtime env recovery action must be commit or rollback")
    with runtime_env_file_lock(path=target):
        state = _load_runtime_env_recovery_state(target)
        if state is None or state["recovery_id"] != recovery_id:
            raise RuntimeEnvFileRecoveryRequired(
                "Runtime env recovery id does not match the durable journal"
            )
        require_runtime_env_file_identity(
            expected_current_identity,
            path=target,
        )
        if action == "rollback":
            return _rollback_runtime_env_recovery_state(
                state,
                outcome="explicit_crash_recovery_rollback",
                error_type=None,
            )
        position, identity = _runtime_env_recovery_target_position(state)
        if position != "published":
            raise RuntimeEnvFileRecoveryRequired(
                "Runtime env recovery commit requires the prepared published bytes"
            )
        committed = {
            **state,
            "phase": "committed",
            "after_identity_hash": identity.get("identity_hash"),
            "resolution": "explicit_crash_recovery_commit",
        }
        _write_runtime_env_recovery_state(committed)
        fact = _record_runtime_env_recovery_audit(
            committed,
            outcome="explicit_crash_recovery_commit",
            error_type=None,
        )
        _cleanup_runtime_env_recovery_files(committed)
        return fact


def normalize_env_file(remove_deprecated: bool = True) -> bool:
    """Atomically normalize the sole root env without creating sidecar copies."""

    with runtime_env_file_lock(path=ENV_PATH):
        _identity, content = _read_runtime_env_snapshot(ENV_PATH)
        if content is None:
            raise RuntimeError("Repository-root .env is missing")
        normalized = _normalized_runtime_env_bytes(
            content,
            remove_deprecated=remove_deprecated,
        )
        if normalized == content:
            return False
        _atomic_replace_runtime_env_bytes(normalized, path=ENV_PATH)
        return True


def _restore_runtime_env_bytes_exact(
    before_bytes: bytes,
    *,
    lock_already_held: bool = False,
) -> None:
    """Atomically restore caller-frozen root bytes after a downstream reject."""

    lock_context = (
        nullcontext()
        if lock_already_held
        else runtime_env_file_lock(path=ENV_PATH)
    )
    with lock_context:
        _current_identity, current_bytes = _read_runtime_env_snapshot(ENV_PATH)
        if current_bytes == before_bytes:
            return
        _atomic_replace_runtime_env_bytes(before_bytes, path=ENV_PATH)


def env_sync_status() -> dict:
    """Check the schema of the sole repository-root runtime ``.env``."""
    actual_keys, bom_keys = _env_keys(ENV_PATH)
    example_keys, _ = _env_keys(ENV_EXAMPLE_PATH)
    declared_managed_keys = {
        key.upper()
        for key in RUNTIME_ENV_SETTINGS
    }
    expected_managed_keys = example_keys.intersection(declared_managed_keys)
    allowed_persisted_keys = example_keys - DEPRECATED_ENV_KEYS
    deprecated_keys = sorted(actual_keys & DEPRECATED_ENV_KEYS)
    active_actual_keys = actual_keys - DEPRECATED_ENV_KEYS
    missing_keys = sorted(expected_managed_keys - active_actual_keys)
    extra_keys = sorted(active_actual_keys - allowed_persisted_keys)
    managed_schema_synced = bool(ENV_PATH.exists()) and not (
        bom_keys or missing_keys or extra_keys
    )
    return {
        "synced": managed_schema_synced,
        "settings_file_present": ENV_PATH.exists(),
        "settings_file_schema_synced": managed_schema_synced,
        "missing_keys": missing_keys,
        "extra_keys": extra_keys,
        "deprecated_keys": deprecated_keys,
        "bom_keys": sorted(set(bom_keys)),
    }


def _runtime_issue(code: str, title: str, message: str, fix_commands: list[str] | None = None) -> dict:
    return {"code": code, "title": title, "message": message, "fix_commands": fix_commands or []}


def _check_postgres() -> bool:
    with suppress(Exception):
        import app.db as db

        with db.SessionLocal() as session:
            session.execute(text("SELECT 1"))
            return True
    return False


def _check_qdrant() -> bool:
    settings = get_settings()
    with suppress(Exception):
        response = httpx.get(f"{settings.qdrant_url.rstrip('/')}/collections", timeout=2.0)
        return response.status_code < 500
    return False


def _check_redis() -> bool:
    settings = get_settings()
    parsed = urlparse(settings.redis_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    with suppress(Exception):
        import socket

        with socket.create_connection((host, port), timeout=2.0) as sock:
            sock.sendall(b"PING\r\n")
            return sock.recv(16).startswith(b"+PONG")
    return False


def _model_bridge_status_with_repair(settings=None) -> dict:
    settings = settings or get_settings()
    if not settings.model_bridge_enabled:
        return {"enabled": False}
    status = model_bridge_status_payload(settings=settings)
    if status.get("reachable") and status.get("admin_available") and not status.get("config_matches"):
        sync_result = sync_model_bridge_runtime_config(settings=settings, raise_on_error=False)
        status = model_bridge_status_payload(settings=settings)
        status["last_sync"] = sync_result
    return status


def _check_model_bridge(status: dict | None = None) -> bool | None:
    settings = get_settings()
    if not settings.model_bridge_enabled:
        return None
    status = status or _model_bridge_status_with_repair(settings=settings)
    return bool(status.get("reachable") and status.get("admin_available") and status.get("config_matches"))


def runtime_check_payload() -> dict:
    env_sync = env_sync_status()
    blocking_issues: list[dict] = []
    warnings: list[dict] = []
    if env_sync["bom_keys"]:
        blocking_issues.append(
            _runtime_issue(
                "env_bom_keys",
                ".env 存在 BOM 前缀键",
                "一个或多个 .env 键名前带有 UTF-8 BOM 前缀。保存设置会先规范化 .env，也可以手动移除对应键名前缀。",
                ["保存一次设置，或手动重写带 BOM 前缀的键名。"],
            )
        )
    if env_sync["missing_keys"] or env_sync["extra_keys"]:
        blocking_issues.append(
            _runtime_issue(
                "env_key_mismatch",
                ".env 与 .env.example 键列表不一致",
                "运行时 .env 参数键列表需要与 .env.example 对齐；检查只比较键名，不比较或暴露具体值。",
                ["对比 .env 和 .env.example，只按需增删键名，不改动密钥值。"],
            )
        )
    settings = get_settings()
    bridge_status = _model_bridge_status_with_repair(settings=settings) if settings.model_bridge_enabled else {"enabled": False}
    infrastructure = {
        "postgres": _check_postgres(),
        "qdrant": _check_qdrant(),
        "redis": _check_redis(),
        "model_bridge": _check_model_bridge(bridge_status),
    }
    for key, ok in infrastructure.items():
        if ok is None:
            continue
        if not ok:
            warnings.append(
                _runtime_issue(
                    f"{key}_unreachable",
                    f"{key} is not reachable",
                    f"The {key} infrastructure check failed from the API process.",
                    [".\\start-app.ps1"],
                )
            )
    if bridge_status.get("enabled"):
        for warning in bridge_status.get("warnings") or []:
            warnings.append(
                _runtime_issue(
                    "model_bridge_config_warning",
                    "Model bridge config needs attention",
                    str(warning),
                    [".\\start-app.ps1"],
                )
            )
    return {
        "env_sync": env_sync,
        "infrastructure": infrastructure,
        "model_bridge_status": bridge_status,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
    }


def _runtime_value_equal(left: Any, right: Any) -> bool:
    if (
        (left is None or left == "" or left == "__none__")
        and (right is None or right == "" or right == "__none__")
    ):
        return True
    if isinstance(left, str) and isinstance(right, str):
        return left.strip() == right.strip()
    if isinstance(left, float) or isinstance(right, float):
        with suppress(TypeError, ValueError):
            return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def _latest_runtime_settings_audit(db: Session, *, for_update: bool = False):
    from app.models import RuntimeSettingsAudit

    statement = select(RuntimeSettingsAudit).order_by(
        RuntimeSettingsAudit.created_at.desc(),
        RuntimeSettingsAudit.id.desc(),
    )
    if for_update:
        statement = statement.with_for_update()
    return db.scalar(statement.limit(1))


def _normalized_settings_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "rq_kmeans_levels" in normalized:
        raise ValueError("rq_kmeans_levels is a read-only fixed protocol value")
    allowed = set(RUNTIME_ENV_SETTINGS) | set(SECRET_CLEAR_CONTROL_KEYS)
    unexpected = sorted(set(normalized) - allowed)
    if unexpected:
        raise ValueError(
            "Unsupported Runtime Settings fields: " + ", ".join(unexpected)
        )
    for key, value in list(normalized.items()):
        if isinstance(value, str):
            normalized[key] = value.strip()
    return normalized


def _validate_provider_targets(
    prospective: Settings,
    requested_keys: set[str],
) -> None:
    provider_specs = {
        "chat": (
            {"chat_base_url", "chat_resolve_ip", "chat_api_protocol"},
            prospective.chat_base_url,
            prospective.chat_resolve_ip,
            prospective.chat_api_protocol,
        ),
        "graph": (
            {"graph_base_url", "graph_resolve_ip", "graph_api_protocol"},
            prospective.graph_base_url,
            prospective.graph_resolve_ip,
            prospective.graph_api_protocol,
        ),
        "embedding": (
            {"embedding_base_url", "embedding_resolve_ip", "embedding_api_protocol"},
            prospective.embedding_base_url,
            prospective.embedding_resolve_ip,
            prospective.embedding_api_protocol,
        ),
    }
    validate_all = "model_bridge_enabled" in requested_keys and bool(
        prospective.model_bridge_enabled
    )
    for purpose, (keys, base_url, resolve_ip, protocol) in provider_specs.items():
        if not validate_all and not requested_keys.intersection(keys):
            continue
        _validated_provider_target(
            base_url=str(base_url or ""),
            resolve_ip=str(resolve_ip or ""),
            protocol=str(protocol or ""),
            purpose=purpose,
        )


def _root_env_updates(
    *,
    prospective: Settings,
    request: Mapping[str, Any],
) -> dict[str, str | int | float | bool | None]:
    updates: dict[str, str | int | float | bool | None] = {}
    for key in sorted(set(request).intersection(RUNTIME_ENV_SETTINGS)):
        if key in SECRET_RUNTIME_SETTING_KEYS or key.upper() in PROCESS_ONLY_ENV_KEYS:
            continue
        value = getattr(prospective, key)
        if isinstance(value, Path):
            value = str(value)
        updates[key] = "" if value is None else value
    for clear_key, secret_key in SECRET_CLEAR_CONTROL_KEYS.items():
        if request.get(clear_key):
            updates[secret_key] = ""
    for secret_key in SECRET_RUNTIME_SETTING_KEYS:
        value = request.get(secret_key)
        if isinstance(value, str) and value.strip():
            updates[secret_key] = value.strip()
    return updates


def _runtime_settings_audit_hash(
    *,
    env_identity_hash: str,
    changed_keys: list[str],
    created_at: str,
) -> str:
    encoded = json.dumps(
        {
            "protocol_version": RUNTIME_SETTINGS_AUDIT_PROTOCOL_VERSION,
            "env_identity_hash": env_identity_hash,
            "changed_keys": sorted(changed_keys),
            "created_at": created_at,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_settings_audit_response_fields(
    row: Any | None,
    *,
    configured_payload: Mapping[str, Any],
) -> dict[str, Any]:
    identity = runtime_env_file_identity(ENV_PATH)
    statuses: dict[str, str] = {}
    if row is not None and row.env_identity_hash == identity.get("identity_hash"):
        statuses = {
            str(key): str(value)
            for key, value in dict(row.field_status_json or {}).items()
            if key not in SECRET_RUNTIME_SETTING_KEYS
        }
        active_settings = get_settings()
        configured_settings = _settings_from_root_env(active_settings)
        for key in set(statuses).intersection(SERVICE_RECREATE_REQUIRED_SETTINGS):
            statuses[key] = (
                "written_and_applied"
                if _runtime_value_equal(
                    getattr(configured_settings, key, configured_payload.get(key)),
                    getattr(active_settings, key, None),
                )
                else "written_pending_service_recreate"
            )
    return {
        "settings_revision": row.version_hash if row is not None else None,
        "setting_statuses": statuses,
        "pending_rebuild_changes": sorted(
            key for key, value in statuses.items()
            if value == "written_pending_rebuild"
        ),
        "pending_service_recreate_changes": sorted(
            key for key, value in statuses.items()
            if value == "written_pending_service_recreate"
        ),
        "pending_hot_changes": sorted(
            key for key, value in statuses.items()
            if value == "written_pending_hot_apply"
        ),
        "settings_file_synced": bool(env_sync_status()["synced"]),
        "apply_error_type": row.last_error_type if row is not None else None,
    }


def runtime_settings_audit_status_payload(
    *,
    configured_payload: Mapping[str, Any],
    db: Session | None = None,
) -> dict[str, Any]:
    if db is not None:
        return _runtime_settings_audit_response_fields(
            _latest_runtime_settings_audit(db),
            configured_payload=configured_payload,
        )
    with suppress(Exception):
        from app.db import SessionLocal

        with SessionLocal() as session:
            return _runtime_settings_audit_response_fields(
                _latest_runtime_settings_audit(session),
                configured_payload=configured_payload,
            )
    return _runtime_settings_audit_response_fields(
        None,
        configured_payload=configured_payload,
    )


def pending_rebuild_setting_keys() -> set[str]:
    """Return pending rebuild key names only; no setting values are mirrored."""

    with suppress(Exception):
        from app.db import SessionLocal

        with SessionLocal() as session:
            row = _latest_runtime_settings_audit(session)
            if row is None:
                return set()
            identity = runtime_env_file_identity(ENV_PATH)
            if row.env_identity_hash != identity.get("identity_hash"):
                return set()
            return {
                str(key)
                for key, value in dict(row.field_status_json or {}).items()
                if value == "written_pending_rebuild"
            }
    return set()


def mark_runtime_settings_rebuild_applied(
    db: Session,
    *,
    changed_keys: list[str],
    runtime_version_hash: str,
) -> None:
    """Close matching rebuild-pending audit fields after atomic promotion."""

    row = _latest_runtime_settings_audit(db, for_update=True)
    if row is None:
        return
    identity = runtime_env_file_identity(ENV_PATH)
    if row.env_identity_hash != identity.get("identity_hash"):
        return
    statuses = dict(row.field_status_json or {})
    for key in set(changed_keys).intersection(REBUILD_REQUIRED_SETTINGS):
        if statuses.get(key) == "written_pending_rebuild":
            statuses[key] = "written_and_applied"
    row.field_status_json = statuses
    row.runtime_version_hash = runtime_version_hash
    row.status = (
        "pending_lifecycle"
        if any(value != "written_and_applied" for value in statuses.values())
        else "applied"
    )
    row.last_error_type = None


def save_model_settings_to_root_env(
    db: Session,
    payload: Mapping[str, Any],
    *,
    source: str = "api",
) -> dict[str, Any]:
    """Write only the root ``.env`` and apply changes by lifecycle class."""

    from app.models import RuntimeSettingsAudit

    request = _normalized_settings_request(payload)
    active_settings = get_settings()
    configured_before = _settings_from_root_env(active_settings)
    requested_keys = set(request).intersection(RUNTIME_ENV_SETTINGS)
    requested_secret_keys = {
        key
        for key in requested_keys.intersection(SECRET_RUNTIME_SETTING_KEYS)
        if isinstance(request.get(key), str) and str(request.get(key) or "").strip()
    }
    requested_secret_keys.update(
        secret_key
        for clear_key, secret_key in SECRET_CLEAR_CONTROL_KEYS.items()
        if request.get(clear_key)
    )

    prospective_model = configured_before.model_dump(mode="python")
    for key in requested_keys - SECRET_RUNTIME_SETTING_KEYS:
        if key.upper() in PROCESS_ONLY_ENV_KEYS:
            continue
        value = request.get(key)
        if key in {"chat_resolve_ip", "graph_resolve_ip", "embedding_resolve_ip"} and not value:
            prospective_model[key] = None
        elif value is not None:
            prospective_model[key] = value
    prospective = Settings.model_validate(prospective_model)
    validate_path_distance_thresholds(
        prospective.agent_path_distance_green_threshold,
        prospective.agent_path_distance_gray_threshold,
        prospective.agent_path_distance_hard_threshold,
    )
    _validate_provider_targets(prospective, requested_keys)
    updates = _root_env_updates(prospective=prospective, request=request)
    current_entries = _env_entries(ENV_PATH)
    changed_keys = [
        key
        for key in requested_keys - SECRET_RUNTIME_SETTING_KEYS
        if key.upper() not in PROCESS_ONLY_ENV_KEYS
        and not _runtime_value_equal(
            getattr(configured_before, key, None),
            getattr(prospective, key, None),
        )
    ]
    for secret_key in requested_secret_keys:
        before_present = bool(
            _deserialize_runtime_env_value(
                current_entries.get(secret_key.upper(), "")
            ).strip()
        )
        clear_requested = any(
            bool(request.get(clear_key)) and mapped_key == secret_key
            for clear_key, mapped_key in SECRET_CLEAR_CONTROL_KEYS.items()
        )
        new_secret = request.get(secret_key)
        after_present = False if clear_requested else (
            True if isinstance(new_secret, str) and bool(new_secret) else before_present
        )
        if before_present != after_present or (
            isinstance(new_secret, str) and bool(new_secret)
        ):
            changed_keys.append(secret_key)
    changed_keys = sorted(set(changed_keys))

    if not ENV_PATH.exists():
        raise RuntimeError("Repository-root .env is missing")

    bridge_keys = {
        "chat_api_protocol", "chat_base_url", "chat_resolve_ip",
        "embedding_api_protocol", "embedding_base_url", "embedding_resolve_ip",
        "model_request_timeout_seconds",
    }
    future_entries = dict(current_entries)
    for key, value in updates.items():
        env_key = key.upper()
        if value is None:
            future_entries.pop(env_key, None)
        else:
            future_entries[env_key] = _serialize_env_value(value)
    hot_changed = sorted(set(changed_keys).intersection(HOT_RELOAD_SETTINGS))
    hot_updates = {key: updates[key] for key in hot_changed if key in updates}
    process_before: dict[str, tuple[bool, str | None]] = {}
    applied_values_before = dict(_RUNTIME_ENV_PROCESS_APPLIED_VALUES)
    bridge_preflight: dict[str, Any] | None = None

    with runtime_env_file_lock(path=ENV_PATH):
        before_identity, before_bytes = _read_runtime_env_snapshot(ENV_PATH)
        if before_bytes is None:
            raise RuntimeError("Repository-root .env is missing")
        if active_settings.model_bridge_enabled and set(changed_keys).intersection(bridge_keys):
            bridge_preflight = preflight_model_bridge_reload(
                settings=active_settings,
                env_entries=future_entries,
            )
        process_before = {
            key.upper(): (key.upper() in os.environ, os.environ.get(key.upper()))
            for key in hot_changed
        }
        publish_bytes = _runtime_env_bytes_with_updates(before_bytes, updates)
        if publish_bytes != before_bytes:
            _atomic_replace_runtime_env_bytes(publish_bytes, path=ENV_PATH)
        try:
            if hot_updates:
                _apply_runtime_env(hot_updates)
                get_settings.cache_clear()
                get_settings()
            if bridge_preflight is not None:
                bridge_result = reload_model_bridge(
                    settings=get_settings(),
                    env_entries=future_entries,
                )
                if bridge_result.get("attempted") is not True or bridge_result.get("ok") is not True:
                    raise RuntimeError("model bridge reload did not commit")
        except Exception:
            _atomic_replace_runtime_env_bytes(before_bytes, path=ENV_PATH)
            for key, (existed, value) in process_before.items():
                if existed:
                    os.environ[key] = str(value or "")
                else:
                    os.environ.pop(key, None)
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES.clear()
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES.update(applied_values_before)
            get_settings.cache_clear()
            if bridge_preflight is not None:
                previous = dict(bridge_preflight["previous"])
                reload_model_bridge(
                    settings=active_settings,
                    env_entries={
                        "CHAT_API_PROTOCOL": str(previous["chat_api_protocol"]),
                        "CHAT_BASE_URL": str(previous["chat_target_base_url"]),
                        "CHAT_RESOLVE_IP": str(previous["chat_resolve_ip"]),
                        "EMBEDDING_API_PROTOCOL": str(previous["embedding_api_protocol"]),
                        "EMBEDDING_BASE_URL": str(previous["embedding_target_base_url"]),
                        "EMBEDDING_RESOLVE_IP": str(previous["embedding_resolve_ip"]),
                        "MODEL_REQUEST_TIMEOUT_SECONDS": str(previous["timeout"]),
                    },
                )
            raise ValueError(
                "Runtime Settings update failed; repository-root .env and current process were restored"
            ) from None

    after_identity = runtime_env_file_identity(ENV_PATH)
    latest = _latest_runtime_settings_audit(db, for_update=True)
    statuses = {
        str(key): str(value)
        for key, value in (
            dict(latest.field_status_json or {}).items() if latest is not None
            and latest.env_identity_hash == before_identity.get("identity_hash") else []
        )
        if value in {
            "written_pending_rebuild",
            "written_pending_service_recreate",
            "written_pending_hot_apply",
        }
    }
    for key in changed_keys:
        if key in SECRET_RUNTIME_SETTING_KEYS:
            continue
        if key in REBUILD_REQUIRED_SETTINGS:
            statuses[key] = (
                "written_and_applied"
                if _runtime_value_equal(
                    getattr(prospective, key, None),
                    getattr(active_settings, key, None),
                )
                else "written_pending_rebuild"
            )
        elif key in SERVICE_RECREATE_REQUIRED_SETTINGS:
            statuses[key] = "written_pending_service_recreate"
        else:
            statuses[key] = "written_and_applied"

    row = None
    if changed_keys:
        created_at = datetime.now(timezone.utc)
        row = RuntimeSettingsAudit(
            protocol_version=RUNTIME_SETTINGS_AUDIT_PROTOCOL_VERSION,
            version_hash=_runtime_settings_audit_hash(
                env_identity_hash=str(after_identity["identity_hash"]),
                changed_keys=changed_keys,
                created_at=created_at.isoformat(),
            ),
            prior_runtime_version_hash=current_runtime_settings_version(),
            changed_keys_json=changed_keys,
            lifecycle_json={
                "hot_reloadable": sorted(set(changed_keys).intersection(HOT_RELOAD_SETTINGS)),
                "rebuild_required": sorted(set(changed_keys).intersection(REBUILD_REQUIRED_SETTINGS)),
                "service_recreate_required": sorted(set(changed_keys).intersection(SERVICE_RECREATE_REQUIRED_SETTINGS)),
            },
            field_status_json=statuses,
            status="written",
            env_identity_hash=str(after_identity["identity_hash"]),
            source=str(source or "api")[:64],
            last_error_type=None,
        )
        db.add(row)
        try:
            db.commit()
            db.refresh(row)
        except Exception:
            db.rollback()
            with runtime_env_file_lock(path=ENV_PATH):
                _atomic_replace_runtime_env_bytes(before_bytes, path=ENV_PATH)
            for key, (existed, value) in process_before.items():
                if existed:
                    os.environ[key] = str(value or "")
                else:
                    os.environ.pop(key, None)
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES.clear()
            _RUNTIME_ENV_PROCESS_APPLIED_VALUES.update(applied_values_before)
            get_settings.cache_clear()
            raise RuntimeError(
                "Runtime Settings audit failed; repository-root .env was restored"
            ) from None

    publish_result: dict[str, Any] | None = None
    if changed_keys:
        publish_result = publish_runtime_settings_version(
            changed_keys=[key.upper() for key in changed_keys],
            source=str(source or "api"),
        )
        delivery_pending = bool(
            publish_result.get("broadcast_pending")
            or publish_result.get("local_refresh_pending")
        )
        if delivery_pending:
            for key in hot_changed:
                if key not in SECRET_RUNTIME_SETTING_KEYS:
                    statuses[key] = "written_pending_hot_apply"
        if row is not None:
            row.field_status_json = statuses
            row.runtime_version_hash = publish_result.get("version_hash")
            row.status = (
                "failed" if delivery_pending else (
                    "pending_lifecycle"
                    if any(value != "written_and_applied" for value in statuses.values())
                    else "applied"
                )
            )
            row.last_error_type = "RuntimeVersionDeliveryPending" if delivery_pending else None
            db.commit()
            db.refresh(row)

    result = model_settings_payload(include_audit_status=False)
    result.update(
        _runtime_settings_audit_response_fields(
            row or latest,
            configured_payload=result,
        )
    )
    result["requires_service_recreate"] = bool(result["pending_service_recreate_changes"])
    result["service_recreate_changes"] = list(result["pending_service_recreate_changes"])
    result["active_mutated"] = bool(hot_changed)
    if publish_result is not None:
        result["runtime_version_broadcast"] = bool(publish_result.get("runtime_version_broadcast"))
        result["runtime_version_broadcast_pending"] = bool(publish_result.get("broadcast_pending"))
        result["runtime_local_refresh_pending"] = bool(publish_result.get("local_refresh_pending"))
    return result


def update_model_settings(payload: dict) -> dict:
    if "rq_kmeans_levels" in payload:
        raise ValueError("rq_kmeans_levels is a read-only fixed protocol value")
    payload = dict(payload)
    active_settings = get_settings()
    for setting_key in ("chat_api_protocol", "graph_api_protocol"):
        if setting_key not in payload:
            continue
        requested_protocol = payload.get(setting_key)
        if (
            not isinstance(requested_protocol, str)
            or requested_protocol not in MODEL_API_PROTOCOL_ALLOWLIST
        ):
            raise ValueError(
                f"{setting_key} must be an allowlisted model API protocol: "
                + ", ".join(sorted(MODEL_API_PROTOCOL_ALLOWLIST))
            )
    if "embedding_api_protocol" in payload:
        requested_embedding_protocol = payload.get("embedding_api_protocol")
        if (
            not isinstance(requested_embedding_protocol, str)
            or requested_embedding_protocol not in EMBEDDING_API_PROTOCOL_ALLOWLIST
        ):
            raise ValueError(
                "embedding_api_protocol must be an allowlisted embedding API protocol: "
                + ", ".join(sorted(EMBEDDING_API_PROTOCOL_ALLOWLIST))
            )
    service_changes: list[str] = []
    for setting_key in sorted(set(payload).intersection(SERVICE_RECREATE_REQUIRED_SETTINGS)):
        requested_value = payload.get(setting_key)
        active_value = getattr(active_settings, setting_key)
        if isinstance(active_value, str) and isinstance(requested_value, str):
            requested_value = requested_value.strip()
        if requested_value != active_value:
            service_changes.append(setting_key)
        payload.pop(setting_key)
    if service_changes:
        # A mixed hot/service request is all-or-nothing.  Returning the
        # recreate requirement before normalization guarantees zero shared-env,
        # singleton, RuntimeSettingsVersion, Redis or model-bridge side effect.
        result = model_settings_payload()
        result.update(
            {
                "requires_service_recreate": True,
                "service_recreate_changes": service_changes,
                "active_mutated": False,
                "runtime_version_broadcast": False,
            }
        )
        return result
    rebuild_protocol_allowlists = {
        "edge_distance_protocol": EDGE_DISTANCE_PROTOCOL_ALLOWLIST,
        "rq_membership_protocol": RQ_MEMBERSHIP_PROTOCOL_ALLOWLIST,
        "edge_projection_protocol": EDGE_PROJECTION_PROTOCOL_ALLOWLIST,
        "edge_type_calibration_protocol": EDGE_TYPE_CALIBRATION_PROTOCOL_ALLOWLIST,
    }
    for setting_key, allowlist in rebuild_protocol_allowlists.items():
        if setting_key not in payload:
            continue
        requested_protocol = payload.get(setting_key)
        if not isinstance(requested_protocol, str) or requested_protocol not in allowlist:
            raise ValueError(
                f"{setting_key} must be a locally allowlisted graph protocol: "
                + ", ".join(sorted(allowlist))
            )
        active_protocol = str(getattr(active_settings, setting_key))
        if requested_protocol != active_protocol:
            raise ValueError(
                "Direct active mutation is forbidden for rebuild-required graph protocol "
                f"setting {setting_key}. Stage a RuntimeSettingsCandidate and complete the "
                "shadow build/evaluation/promotion lifecycle instead."
            )
        # The active settings endpoint may acknowledge an exact identity echo, but it must
        # never write or broadcast rebuild-required protocol settings as a hot reload.
        payload.pop(setting_key)
    rebuild_numeric_specs: dict[str, tuple[str, float, float]] = {
        "rq_membership_temperature": ("float", 0.0, 10.0),
        "rq_membership_top_m": ("int", 1.0, 6.0),
        "rq_membership_probability_threshold": ("float", 0.0, 1.0),
    }
    for setting_key, (value_type, lower, upper) in rebuild_numeric_specs.items():
        if setting_key not in payload:
            continue
        requested_value = payload.get(setting_key)
        if value_type == "int":
            if type(requested_value) is not int or not lower <= requested_value <= upper:
                raise ValueError(
                    f"{setting_key} must be an integer between {int(lower)} and {int(upper)}"
                )
            normalized_value: int | float = requested_value
        else:
            if (
                isinstance(requested_value, bool)
                or not isinstance(requested_value, (int, float))
                or not math.isfinite(float(requested_value))
                or not lower <= float(requested_value) <= upper
                or (setting_key == "rq_membership_temperature" and float(requested_value) <= 0.0)
            ):
                interval = "(0, 10]" if setting_key == "rq_membership_temperature" else "[0, 1]"
                raise ValueError(f"{setting_key} must be a finite number in {interval}")
            normalized_value = float(requested_value)
        active_value = getattr(active_settings, setting_key)
        if normalized_value != active_value:
            raise ValueError(
                "Direct active mutation is forbidden for rebuild-required RQ membership "
                f"setting {setting_key}. Stage a RuntimeSettingsCandidate and complete the "
                "shadow build/evaluation/promotion lifecycle instead."
            )
        payload.pop(setting_key)
    rebuild_changes: list[str] = []
    for setting_key in sorted(set(payload).intersection(REBUILD_REQUIRED_SETTINGS)):
        requested_value = payload.get(setting_key)
        active_value = getattr(active_settings, setting_key)
        if isinstance(active_value, str) and isinstance(requested_value, str):
            requested_value = requested_value.strip()
        if requested_value != active_value:
            rebuild_changes.append(setting_key)
        payload.pop(setting_key)
    if rebuild_changes:
        raise ValueError(
            "Direct active mutation is forbidden for rebuild-required Runtime Settings: "
            + ", ".join(rebuild_changes)
            + ". Stage a RuntimeSettingsCandidate and complete dry-run, bounded shadow "
            "parse/rechunk/vector/four-layer build, measured evaluation and promotion."
        )
    if not payload:
        return model_settings_payload()
    if "gray_zone_rule_protocol" in payload:
        requested_gray_zone_protocol = payload.get("gray_zone_rule_protocol")
        if (
            not isinstance(requested_gray_zone_protocol, str)
            or requested_gray_zone_protocol not in GRAY_ZONE_RULE_PROTOCOL_ALLOWLIST
        ):
            raise ValueError(
                "gray_zone_rule_protocol must be an allowlisted deterministic local protocol: "
                + ", ".join(sorted(GRAY_ZONE_RULE_PROTOCOL_ALLOWLIST))
            )
    if "gray_zone_observation_cadence" in payload:
        requested_observation_cadence = payload.get("gray_zone_observation_cadence")
        if (
            type(requested_observation_cadence) is not int
            or not 1 <= requested_observation_cadence <= GRAY_ZONE_OBSERVATION_CADENCE_MAX
        ):
            raise ValueError(
                "gray_zone_observation_cadence must be an integer between 1 and "
                f"{GRAY_ZONE_OBSERVATION_CADENCE_MAX}"
            )
    if "traversal_observation_budget" in payload:
        requested_observation_budget = payload.get("traversal_observation_budget")
        if (
            type(requested_observation_budget) is not int
            or not 1 <= requested_observation_budget <= TRAVERSAL_OBSERVATION_BUDGET_MAX
        ):
            raise ValueError(
                "traversal_observation_budget must be an integer between 1 and "
                f"{TRAVERSAL_OBSERVATION_BUDGET_MAX}"
            )
    prospective_path_thresholds: dict[str, float] = {}
    for threshold_key in (
        "agent_path_distance_green_threshold",
        "agent_path_distance_gray_threshold",
        "agent_path_distance_hard_threshold",
    ):
        threshold_value = (
            payload.get(threshold_key)
            if threshold_key in payload
            else getattr(active_settings, threshold_key)
        )
        if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
            raise ValueError(f"{threshold_key} must be a finite number")
        prospective_path_thresholds[threshold_key] = float(threshold_value)
    validate_path_distance_thresholds(
        prospective_path_thresholds["agent_path_distance_green_threshold"],
        prospective_path_thresholds["agent_path_distance_gray_threshold"],
        prospective_path_thresholds["agent_path_distance_hard_threshold"],
    )
    rebuild_required_vector_changes: list[str] = []
    if "embedding_model" in payload and payload.get("embedding_model") is not None:
        requested_model = str(payload.get("embedding_model") or "").strip()
        if requested_model != active_settings.embedding_model:
            rebuild_required_vector_changes.append("embedding_model")
    if "embedding_dimensions" in payload and payload.get("embedding_dimensions") is not None:
        requested_dimension = payload.get("embedding_dimensions")
        if type(requested_dimension) is not int or requested_dimension <= 0:
            raise ValueError("embedding_dimensions must be a positive integer")
        if requested_dimension != active_settings.embedding_dimensions:
            rebuild_required_vector_changes.append("embedding_dimensions")
    if rebuild_required_vector_changes:
        raise ValueError(
            "Direct active mutation is forbidden for rebuild-required vector settings: "
            + ", ".join(sorted(rebuild_required_vector_changes))
            + ". Stage a RuntimeSettingsCandidate and complete the shadow build/evaluation/"
            "promotion lifecycle instead."
        )
    # Shared runtime bytes are a side effect.  Keep normalization behind all
    # lifecycle, bridge-target and bridge-admin validation gates so an invalid
    # request cannot mutate the managed environment before failing closed.
    normalized = False
    updates: dict[str, str | int | float | bool | None] = {}
    key_map = {
        "chat_api_protocol": "chat_api_protocol",
        "chat_base_url": "chat_base_url",
        "graph_api_protocol": "graph_api_protocol",
        "embedding_api_protocol": "embedding_api_protocol",
        "graph_base_url": "graph_base_url",
        "embedding_base_url": "embedding_base_url",
        "model_bridge_enabled": "model_bridge_enabled",
        "chat_resolve_ip": "chat_resolve_ip",
        "graph_resolve_ip": "graph_resolve_ip",
        "embedding_resolve_ip": "embedding_resolve_ip",
        "embedding_model": "embedding_model",
        "chat_model": "chat_model",
        "graph_model": "graph_model",
        "embedding_dimensions": "embedding_dimensions",
        "embedding_batch_size": "embedding_batch_size",
        "worker_concurrency": "worker_concurrency",
        "model_request_concurrency": "model_request_concurrency",
        "model_request_timeout_seconds": "model_request_timeout_seconds",
        "chat_json_max_tokens": "chat_json_max_tokens",
        "agent_request_concurrency": "agent_request_concurrency",
        "source_io_concurrency": "source_io_concurrency",
        "agent_request_queue_limit": "agent_request_queue_limit",
        "agent_request_queue_timeout_seconds": "agent_request_queue_timeout_seconds",
        "agent_request_lease_ttl_seconds": "agent_request_lease_ttl_seconds",
        "upload_max_bytes": "upload_max_bytes",
        "concept_i18n_enabled": "concept_i18n_enabled",
        "query_facet_bilingual_enabled": "query_facet_bilingual_enabled",
        "query_facet_posterior_enabled": "query_facet_posterior_enabled",
        "query_facet_posterior_observation_budget": (
            "query_facet_posterior_observation_budget"
        ),
        "query_facet_posterior_round_budget": (
            "query_facet_posterior_round_budget"
        ),
        "query_facet_posterior_convergence_epsilon": (
            "query_facet_posterior_convergence_epsilon"
        ),
        "enable_auto_tpe": "enable_auto_tpe",
        "fixed_chunk_size_tokens": "fixed_chunk_size_tokens",
        "fixed_chunk_overlap_tokens": "fixed_chunk_overlap_tokens",
        "context_package_token_budget": "context_package_token_budget",
        "mid_concept_extraction_max_model_batches": "mid_concept_extraction_max_model_batches",
        "mid_concept_extraction_max_candidates_per_batch": "mid_concept_extraction_max_candidates_per_batch",
        "mid_concept_extraction_max_tokens_per_batch": "mid_concept_extraction_max_tokens_per_batch",
        "mid_concept_candidate_keep_threshold": "mid_concept_candidate_keep_threshold",
        "rq_kmeans_max_k": "rq_kmeans_max_k",
        "rq_residual_tau": "rq_residual_tau",
        "dense_knn_k_min": "dense_knn_k_min",
        "dense_knn_k_max": "dense_knn_k_max",
        "dense_reverse_b_min_base": "dense_reverse_b_min_base",
        "dense_reverse_b_max_base": "dense_reverse_b_max_base",
        "dense_reverse_b_min_doc": "dense_reverse_b_min_doc",
        "dense_reverse_b_max_doc": "dense_reverse_b_max_doc",
        "dense_reverse_b_min_lang": "dense_reverse_b_min_lang",
        "dense_reverse_b_max_lang": "dense_reverse_b_max_lang",
        "dense_min_cosine": "dense_min_cosine",
        "dense_strong_cosine": "dense_strong_cosine",
        "cross_doc_out_quota_min": "cross_doc_out_quota_min",
        "cross_doc_out_quota_max": "cross_doc_out_quota_max",
        "cross_doc_min_cosine": "cross_doc_min_cosine",
        "cross_language_out_quota_min": "cross_language_out_quota_min",
        "cross_language_out_quota_max": "cross_language_out_quota_max",
        "cross_language_min_cosine": "cross_language_min_cosine",
        "tpe_trial_budget": "tpe_trial_budget",
        "tpe_startup_random_trials": "tpe_startup_random_trials",
        "tpe_good_quantile_gamma": "tpe_good_quantile_gamma",
        "tpe_probe_query_budget": "tpe_probe_query_budget",
        "tpe_trial_timeout_seconds": "tpe_trial_timeout_seconds",
        "tpe_candidate_pool_size": "tpe_candidate_pool_size",
        "operating_point_hard_gate_max_edge_density": "operating_point_hard_gate_max_edge_density",
        "operating_point_hard_gate_max_isolated_ratio": "operating_point_hard_gate_max_isolated_ratio",
        "operating_point_hard_gate_max_hubness_ratio": "operating_point_hard_gate_max_hubness_ratio",
        "operating_point_hard_gate_min_structure_recovery_rate": "operating_point_hard_gate_min_structure_recovery_rate",
        "operating_point_hard_gate_max_candidate_latency_p95_ms": "operating_point_hard_gate_max_candidate_latency_p95_ms",
        "retrieval_result_top_k_default": "retrieval_result_top_k_default",
        "agent_coarse_initial_budget": "agent_coarse_initial_budget",
        "agent_coarse_total_budget": "agent_coarse_total_budget",
        "agent_coarse_top_k": "agent_coarse_top_k",
        "agent_mid_per_coarse_budget": "agent_mid_per_coarse_budget",
        "agent_coarse_drilldown_mid_initial_budget": "agent_coarse_drilldown_mid_initial_budget",
        "agent_mid_initial_budget": "agent_mid_initial_budget",
        "agent_mid_top_k": "agent_mid_top_k",
        "agent_chunk_per_mid_budget": "agent_chunk_per_mid_budget",
        "agent_chunk_initial_budget": "agent_chunk_initial_budget",
        "agent_chunk_top_k": "agent_chunk_top_k",
        "candidate_pool_dedupe_budget": "candidate_pool_dedupe_budget",
        "agent_max_depth_per_layer": "agent_max_depth_per_layer",
        "agent_max_labels_per_node": "agent_max_labels_per_node",
        "agent_max_edge_reuse": "agent_max_edge_reuse",
        "agent_max_cycle_reward_per_path": "agent_max_cycle_reward_per_path",
        "agent_cycle_reward_distance_threshold": "agent_cycle_reward_distance_threshold",
        "agent_path_distance_green_threshold": "agent_path_distance_green_threshold",
        "agent_path_distance_gray_threshold": "agent_path_distance_gray_threshold",
        "agent_path_distance_hard_threshold": "agent_path_distance_hard_threshold",
        "gray_zone_rule_protocol": "gray_zone_rule_protocol",
        "gray_zone_observation_cadence": "gray_zone_observation_cadence",
        "traversal_observation_budget": "traversal_observation_budget",
        "agent_structure_restore_per_chunk_budget": "agent_structure_restore_per_chunk_budget",
        "agent_structure_restore_budget": "agent_structure_restore_budget",
        "context_path_summary_budget": "context_path_summary_budget",
        "agent_planning_round_budget": "agent_planning_round_budget",
        "agent_max_typed_actions_per_round": "agent_max_typed_actions_per_round",
        "agent_repair_round_budget": "agent_repair_round_budget",
        "agent_verification_budget": "agent_verification_budget",
    }
    nullable_setting_keys: set[str] = set()
    for key, env_key in key_map.items():
        if key not in payload:
            continue
        value = payload.get(key)
        if key in {"chat_resolve_ip", "graph_resolve_ip", "embedding_resolve_ip"} and (value is None or (isinstance(value, str) and not value.strip())):
            updates[env_key] = ""
        elif key in nullable_setting_keys and value is None:
            updates[env_key] = None
        elif value is not None:
            updates[env_key] = (
                value
                if key in {"chat_base_url", "graph_base_url", "embedding_base_url"}
                else (value.strip() if isinstance(value, str) else value)
            )

    chat_api_key = payload.get("chat_api_key")
    if payload.get("clear_chat_api_key"):
        updates["chat_api_key"] = ""
    elif isinstance(chat_api_key, str) and chat_api_key.strip():
        updates["chat_api_key"] = chat_api_key.strip()

    graph_api_key = payload.get("graph_api_key")
    if payload.get("clear_graph_api_key"):
        updates["graph_api_key"] = ""
    elif isinstance(graph_api_key, str) and graph_api_key.strip():
        updates["graph_api_key"] = graph_api_key.strip()

    embedding_api_key = payload.get("embedding_api_key")
    if payload.get("clear_embedding_api_key"):
        updates["embedding_api_key"] = ""
    elif isinstance(embedding_api_key, str) and embedding_api_key.strip():
        updates["embedding_api_key"] = embedding_api_key.strip()

    bridge_reload_result: dict | None = None
    bridge_preflight: dict[str, Any] | None = None
    future_settings = None
    runtime_before_bytes: bytes | None = None
    process_before: dict[str, tuple[bool, str | None]] = {}
    applied_values_before = dict(_RUNTIME_ENV_PROCESS_APPLIED_VALUES)
    if updates:
        future_env_entries = dict(_env_entries(ENV_PATH))
        for key, value in updates.items():
            env_key = key.upper()
            if value is None:
                future_env_entries.pop(env_key, None)
            else:
                future_env_entries[env_key] = _serialize_env_value(value)

        def future_value(env_key: str, fallback: object) -> str:
            if env_key in future_env_entries:
                return _deserialize_runtime_env_value(future_env_entries[env_key])
            return str(fallback or "")

        changed_env_keys = {key.upper() for key in updates}
        provider_groups = {
            "chat": {
                "CHAT_API_PROTOCOL",
                "CHAT_BASE_URL",
                "CHAT_RESOLVE_IP",
                "CHAT_API_KEY",
                "CHAT_MODEL",
            },
            "graph": {
                "GRAPH_API_PROTOCOL",
                "GRAPH_BASE_URL",
                "GRAPH_RESOLVE_IP",
                "GRAPH_API_KEY",
                "GRAPH_MODEL",
            },
            "embedding": {
                "EMBEDDING_API_PROTOCOL",
                "EMBEDDING_BASE_URL",
                "EMBEDDING_RESOLVE_IP",
                "EMBEDDING_API_KEY",
                "EMBEDDING_MODEL",
            },
        }
        if changed_env_keys & provider_groups["chat"]:
            try:
                _validated_provider_target(
                    base_url=future_value("CHAT_BASE_URL", active_settings.chat_base_url),
                    resolve_ip=future_value(
                        "CHAT_RESOLVE_IP", active_settings.chat_resolve_ip
                    ),
                    protocol=future_value(
                        "CHAT_API_PROTOCOL", active_settings.chat_api_protocol
                    ),
                    purpose="chat",
                )
            except ValueError as exc:
                raise ValueError(
                    "CHAT_BASE_URL/CHAT_RESOLVE_IP/CHAT_API_PROTOCOL "
                    f"configuration is invalid: {exc}"
                ) from None
        if changed_env_keys & provider_groups["graph"]:
            try:
                _validated_provider_target(
                    base_url=future_value(
                        "GRAPH_BASE_URL", active_settings.graph_base_url
                    ),
                    resolve_ip=future_value(
                        "GRAPH_RESOLVE_IP", active_settings.graph_resolve_ip
                    ),
                    protocol=future_value(
                        "GRAPH_API_PROTOCOL", active_settings.graph_api_protocol
                    ),
                    purpose="graph",
                )
            except ValueError as exc:
                raise ValueError(
                    "GRAPH_BASE_URL/GRAPH_RESOLVE_IP/GRAPH_API_PROTOCOL "
                    f"configuration is invalid: {exc}"
                ) from None
        if changed_env_keys & provider_groups["embedding"]:
            try:
                _validated_provider_target(
                    base_url=future_value(
                        "EMBEDDING_BASE_URL", active_settings.embedding_base_url
                    ),
                    resolve_ip=future_value(
                        "EMBEDDING_RESOLVE_IP", active_settings.embedding_resolve_ip
                    ),
                    protocol=future_value(
                        "EMBEDDING_API_PROTOCOL",
                        active_settings.embedding_api_protocol,
                    ),
                    purpose="embedding",
                )
            except ValueError as exc:
                raise ValueError(
                    "EMBEDDING_API_PROTOCOL/EMBEDDING_BASE_URL/"
                    "EMBEDDING_RESOLVE_IP configuration is "
                    f"invalid: {exc}"
                ) from None
        future_bridge_enabled = future_value(
            "MODEL_BRIDGE_ENABLED", active_settings.model_bridge_enabled
        ).strip().lower() in {
            "true",
            "1",
            "yes",
            "on",
        }
        bridge_config_keys = {
            "CHAT_API_PROTOCOL",
            "CHAT_BASE_URL",
            "CHAT_RESOLVE_IP",
            "EMBEDDING_API_PROTOCOL",
            "EMBEDDING_BASE_URL",
            "EMBEDDING_RESOLVE_IP",
            "MODEL_REQUEST_TIMEOUT_SECONDS",
            "MODEL_BRIDGE_ENABLED",
        }
        if future_bridge_enabled and changed_env_keys & bridge_config_keys:
            try:
                desired = _validated_desired_bridge_config(
                    _desired_bridge_config(get_settings(), future_env_entries)
                )
                _validated_bridge_admin_token(
                    get_settings().model_bridge_admin_token
                )
            except (ValueError, RuntimeError):
                raise ValueError(
                    "MODEL_BRIDGE_ENABLED=true requires valid HTTPS/public "
                    "CHAT_BASE_URL and EMBEDDING_BASE_URL targets, public-unicast "
                    "CHAT_RESOLVE_IP/EMBEDDING_RESOLVE_IP values when configured, "
                    "and an explicit non-default MODEL_BRIDGE_ADMIN_TOKEN"
                ) from None
            self_target_keys = _bridge_self_target_keys(desired, get_settings())
            if self_target_keys:
                raise ValueError(
                    "MODEL_BRIDGE_ENABLED=true 时 CHAT_BASE_URL/EMBEDDING_BASE_URL 必须是真实模型服务地址，不能指向模型桥自身: "
                    + ", ".join(self_target_keys)
                )

            future_settings = active_settings.model_copy(
                update={"model_bridge_enabled": True}
            )
            bridge_preflight = {"pending": True}
            process_before = {
                key: (key in os.environ, os.environ.get(key))
                for key in changed_env_keys
            }

    runtime_publish_result: dict[str, Any] | None = None

    def publish_after_update(*, normalized: bool) -> None:
        nonlocal runtime_publish_result
        if not normalized and not updates:
            return
        changed_keys = [key.upper() for key in updates]
        if normalized:
            changed_keys.append("ENV_FILE_NORMALIZED")
        from app.services.cache_manager import clear_cache_manager

        clear_cache_manager()
        runtime_publish_result = publish_runtime_settings_version(
            changed_keys=changed_keys,
            source="api",
        )

    if bridge_preflight is not None:
        with runtime_env_file_lock(path=ENV_PATH):
            before_identity, runtime_before_bytes = _read_runtime_env_snapshot(
                ENV_PATH
            )
            if runtime_before_bytes is None:
                raise ValueError(
                    "Model bridge runtime update requires an existing managed env file"
                )
            bridge_preflight = preflight_model_bridge_reload(
                settings=future_settings,
                env_entries=future_env_entries,
            )
            try:
                normalized_bytes = _normalized_runtime_env_bytes(
                    runtime_before_bytes,
                    remove_deprecated=True,
                )
                normalized = normalized_bytes != runtime_before_bytes
                if normalized:
                    _atomic_replace_runtime_env_bytes(
                        normalized_bytes,
                        path=ENV_PATH,
                    )
                _update_env_file(updates, lock_already_held=True)
                _apply_runtime_env(updates)
                get_settings.cache_clear()
                bridge_reload_result = reload_model_bridge(
                    settings=future_settings,
                    env_entries=future_env_entries,
                )
                if (
                    bridge_reload_result.get("attempted") is not True
                    or bridge_reload_result.get("ok") is not True
                ):
                    raise RuntimeError("model bridge reload did not commit")
                publish_after_update(normalized=normalized)
            except Exception as update_error:
                _restore_runtime_env_bytes_exact(
                    runtime_before_bytes,
                    lock_already_held=True,
                )
                for key, (existed, value) in process_before.items():
                    if existed:
                        os.environ[key] = str(value or "")
                    else:
                        os.environ.pop(key, None)
                _RUNTIME_ENV_PROCESS_APPLIED_VALUES.clear()
                _RUNTIME_ENV_PROCESS_APPLIED_VALUES.update(applied_values_before)
                get_settings.cache_clear()
                from app.services.cache_manager import clear_cache_manager

                clear_cache_manager()
                previous = dict(bridge_preflight["previous"])
                rollback_entries = {
                    "CHAT_API_PROTOCOL": str(previous["chat_api_protocol"]),
                    "CHAT_BASE_URL": str(previous["chat_target_base_url"]),
                    "CHAT_RESOLVE_IP": str(previous["chat_resolve_ip"]),
                    "EMBEDDING_API_PROTOCOL": str(
                        previous["embedding_api_protocol"]
                    ),
                    "EMBEDDING_BASE_URL": str(
                        previous["embedding_target_base_url"]
                    ),
                    "EMBEDDING_RESOLVE_IP": str(
                        previous["embedding_resolve_ip"]
                    ),
                    "MODEL_REQUEST_TIMEOUT_SECONDS": str(previous["timeout"]),
                }
                rollback_result = reload_model_bridge(
                    settings=active_settings.model_copy(
                        update={"model_bridge_enabled": True}
                    ),
                    env_entries=rollback_entries,
                )
                if (
                    rollback_result.get("attempted") is not True
                    or rollback_result.get("ok") is not True
                ):
                    raise RuntimeError(
                        "Runtime settings rejected and root env bytes were restored, "
                        "but model bridge rollback failed"
                    ) from update_error
                raise ValueError(
                    "Runtime settings update was rejected; exact env bytes and model "
                    "bridge configuration were restored"
                ) from None
    else:
        normalized = normalize_env_file()
        if updates:
            _update_env_file(updates)
            _apply_runtime_env(updates)
            get_settings.cache_clear()
        publish_after_update(normalized=normalized)
    result = model_settings_payload()
    if runtime_publish_result is not None:
        result["runtime_version_broadcast"] = bool(
            runtime_publish_result.get("runtime_version_broadcast")
        )
        result["runtime_version_broadcast_pending"] = bool(
            runtime_publish_result.get("broadcast_pending")
        )
        result["runtime_local_refresh_pending"] = bool(
            runtime_publish_result.get("local_refresh_pending")
        )
    if bridge_reload_result is not None:
        status = dict(result.get("model_bridge_status") or {})
        status["last_reload"] = bridge_reload_result
        if bridge_reload_result.get("attempted") and not bridge_reload_result.get("ok"):
            status.setdefault("warnings", [])
            status["warnings"].append(f"Model bridge reload failed: {bridge_reload_result.get('error') or bridge_reload_result.get('status_code')}")
        result["model_bridge_status"] = status
    return result


def initialize_runtime_env_from_root_file() -> dict[str, Any]:
    """Read and apply the repository-root runtime authority without writing it.

    Importing this module is intentionally side-effect free.  API, worker and
    beat entry points call this function after their earliest platform safety
    gate and before constructing settings-dependent clients.  Missing bytes or
    a pending recovery journal fail closed.  Bootstrap/normalization remains an
    explicit operator action and is never performed by a service import.
    """

    identity = runtime_env_file_identity(ENV_PATH)
    if not identity.get("exists"):
        raise RuntimeError(
            "Repository-root .env is missing; restore it before starting API or workers"
        )
    _apply_runtime_env_file_to_process_environment()
    get_settings.cache_clear()
    return {
        "initialized": True,
        "reason": "root_runtime_env_applied",
        "bootstrap": None,
        "identity_hash": identity["identity_hash"],
    }

from __future__ import annotations

import os
import hashlib
import json
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from sqlalchemy import text

from app.core.config import WORKSPACE_ROOT, get_settings
from app.services.error_sanitizer import public_exception_message, sanitize_error_message


ENV_PATH = WORKSPACE_ROOT / ".env"
ENV_EXAMPLE_PATH = WORKSPACE_ROOT / ".env.example"
SETTINGS_VERSION_REDIS_KEY = "symbograph:runtime_settings:version"
SETTINGS_VERSION_CHANNEL = "symbograph:runtime_settings:updated"
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


def read_env_int(key: str, default: int = 0) -> int:
    """直接从 .env 文件读取整数值（热加载，绕过 os.environ 缓存）。"""
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def read_env_float(key: str, default: float = 0.0) -> float:
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_env_bool(key: str, default: bool = False) -> bool:
    value = _env_entries(ENV_PATH).get(key.upper())
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def runtime_lifecycle_payload() -> dict:
    rebuild_required = [
        "fixed_chunk_size_tokens",
        "fixed_chunk_overlap_tokens",
        "embedding_model",
        "embedding_dimensions",
        "graph_base_url",
        "graph_resolve_ip",
        "graph_model",
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
        "rq_kmeans_levels",
        "rq_kmeans_max_k",
        "rq_residual_tau",
        "mid_concept_extraction_max_model_batches",
        "mid_concept_extraction_max_candidates_per_batch",
        "mid_concept_extraction_max_tokens_per_batch",
        "mid_concept_candidate_keep_threshold",
    ]
    hot_reloadable = [
        "chat_base_url",
        "chat_resolve_ip",
        "embedding_base_url",
        "embedding_resolve_ip",
        "embedding_batch_size",
        "model_request_concurrency",
        "model_request_timeout_seconds",
        "concept_i18n_enabled",
        "enable_auto_tpe",
        "tpe_trial_budget",
        "tpe_startup_random_trials",
        "tpe_good_quantile_gamma",
        "tpe_probe_query_budget",
        "tpe_trial_timeout_seconds",
        "tpe_candidate_pool_size",
        "operating_point_hard_gate_max_edge_density",
        "operating_point_hard_gate_max_isolated_ratio",
        "operating_point_hard_gate_max_hubness_ratio",
        "operating_point_hard_gate_min_structure_recovery_rate",
        "operating_point_hard_gate_max_candidate_latency_p95_ms",
        "context_package_token_budget",
        "agent_coarse_total_budget",
        "agent_mid_per_coarse_budget",
        "agent_mid_top_k",
        "agent_chunk_per_mid_budget",
        "agent_chunk_top_k",
        "agent_max_depth_per_layer",
        "agent_max_labels_per_node",
        "agent_max_edge_reuse",
        "agent_max_cycle_reward_per_path",
        "agent_cycle_reward_distance_threshold",
        "agent_path_distance_green_threshold",
        "agent_path_distance_gray_threshold",
        "agent_path_distance_hard_threshold",
        "candidate_pool_dedupe_budget",
        "agent_structure_restore_budget",
        "context_path_summary_budget",
        "agent_planning_round_budget",
        "agent_max_typed_actions_per_round",
        "agent_repair_round_budget",
        "agent_verification_budget",
    ]
    service_recreate_required = [
        "worker_concurrency",
        "model_bridge_enabled",
        "model_bridge_port",
    ]
    return {
        "hot_reloadable": hot_reloadable,
        "rebuild_required": rebuild_required,
        "service_recreate_required": service_recreate_required,
        "candidate_version_required_for": rebuild_required,
        "operating_point_gate": {
            "required": False,
            "stages": ["automatic_lightweight_tpe", "single_active_relation_graph_write"],
            "hard_gates": ["edge_density", "isolated_ratio", "hubness", "structure_recovery", "candidate_latency"],
        },
        "redaction": {
            "secret_fields": ["chat_api_key", "graph_api_key", "embedding_api_key", "model_bridge_admin_token", "api_keys"],
            "payload_exposes_secret_values": False,
        },
    }


def model_settings_payload() -> dict:
    settings = get_settings()
    env_entries = _env_entries(ENV_PATH)
    chat_base_url = env_entries.get("CHAT_BASE_URL", "" if settings.model_bridge_enabled else settings.chat_base_url)
    graph_base_url = env_entries.get("GRAPH_BASE_URL", settings.graph_base_url)
    embedding_base_url = env_entries.get("EMBEDDING_BASE_URL", "" if settings.model_bridge_enabled else settings.embedding_base_url)
    model_bridge_enabled = settings.model_bridge_enabled or env_entries.get("MODEL_BRIDGE_ENABLED", "false").lower() == "true"
    model_bridge_status = model_bridge_status_payload(settings=settings, env_entries=env_entries)
    return {
        "provider": "openai_compatible",
        "chat_base_url": chat_base_url,
        "graph_base_url": graph_base_url,
        "embedding_base_url": embedding_base_url,
        "effective_chat_base_url": settings.chat_base_url,
        "effective_graph_base_url": settings.graph_base_url,
        "effective_embedding_base_url": settings.embedding_base_url,
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
        "rq_kmeans_levels": read_env_int("RQ_KMEANS_LEVELS", settings.rq_kmeans_levels),
        "rq_kmeans_max_k": read_env_int("RQ_KMEANS_MAX_K", settings.rq_kmeans_max_k),
        "rq_residual_tau": read_env_float("RQ_RESIDUAL_TAU", settings.rq_residual_tau),
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
        "agent_coarse_total_budget": read_env_int("AGENT_COARSE_TOTAL_BUDGET", settings.agent_coarse_total_budget),
        "agent_mid_per_coarse_budget": read_env_int("AGENT_MID_PER_COARSE_BUDGET", settings.agent_mid_per_coarse_budget),
        "agent_mid_top_k": read_env_int("AGENT_MID_TOP_K", settings.agent_mid_top_k),
        "agent_chunk_per_mid_budget": read_env_int("AGENT_CHUNK_PER_MID_BUDGET", settings.agent_chunk_per_mid_budget),
        "agent_chunk_top_k": read_env_int("AGENT_CHUNK_TOP_K", settings.agent_chunk_top_k),
        "agent_max_depth_per_layer": read_env_int("AGENT_MAX_DEPTH_PER_LAYER", settings.agent_max_depth_per_layer),
        "agent_max_labels_per_node": read_env_int("AGENT_MAX_LABELS_PER_NODE", settings.agent_max_labels_per_node),
        "agent_max_edge_reuse": read_env_int("AGENT_MAX_EDGE_REUSE", settings.agent_max_edge_reuse),
        "agent_max_cycle_reward_per_path": read_env_float("AGENT_MAX_CYCLE_REWARD_PER_PATH", settings.agent_max_cycle_reward_per_path),
        "agent_cycle_reward_distance_threshold": read_env_float("AGENT_CYCLE_REWARD_DISTANCE_THRESHOLD", settings.agent_cycle_reward_distance_threshold),
        "agent_path_distance_green_threshold": read_env_float("AGENT_PATH_DISTANCE_GREEN_THRESHOLD", settings.agent_path_distance_green_threshold),
        "agent_path_distance_gray_threshold": read_env_float("AGENT_PATH_DISTANCE_GRAY_THRESHOLD", settings.agent_path_distance_gray_threshold),
        "agent_path_distance_hard_threshold": read_env_float("AGENT_PATH_DISTANCE_HARD_THRESHOLD", settings.agent_path_distance_hard_threshold),
        "candidate_pool_dedupe_budget": read_env_int("CANDIDATE_POOL_DEDUPE_BUDGET", settings.candidate_pool_dedupe_budget),
        "agent_structure_restore_budget": read_env_int("AGENT_STRUCTURE_RESTORE_BUDGET", settings.agent_structure_restore_budget),
        "context_path_summary_budget": read_env_int("CONTEXT_PATH_SUMMARY_BUDGET", settings.context_path_summary_budget),
        "agent_planning_round_budget": read_env_int("AGENT_PLANNING_ROUND_BUDGET", settings.agent_planning_round_budget),
        "agent_max_typed_actions_per_round": read_env_int("AGENT_MAX_TYPED_ACTIONS_PER_ROUND", settings.agent_max_typed_actions_per_round),
        "agent_repair_round_budget": read_env_int("AGENT_REPAIR_ROUND_BUDGET", settings.agent_repair_round_budget),
        "agent_verification_budget": read_env_int("AGENT_VERIFICATION_BUDGET", settings.agent_verification_budget),
        "concept_i18n_enabled": read_env_bool("CONCEPT_I18N_ENABLED", settings.concept_i18n_enabled),
        "enable_model_fallback": settings.enable_model_fallback,
        "enable_database_fallback": settings.enable_database_fallback,
        "has_chat_api_key": bool(settings.chat_api_key),
        "has_graph_api_key": bool(settings.graph_api_key),
        "degraded_mode": not settings.chat_api_key or not settings.embedding_api_key or not settings.embedding_base_url,
        "embedding_resolve_ip": env_entries.get("EMBEDDING_RESOLVE_IP", "" if settings.model_bridge_enabled else (settings.embedding_resolve_ip or "")),
        "has_embedding_api_key": bool(settings.embedding_api_key),
        "model_bridge_status": model_bridge_status,
        "runtime_settings_version": current_runtime_settings_version(),
        "lifecycle": runtime_lifecycle_payload(),
    }


def _serialize_env_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if not text:
        return ""
    if any(char.isspace() for char in text) or any(char in text for char in ['"', "#", "="]):
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return text


def _update_env_file(updates: dict[str, str | int | float | bool | None]) -> None:
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    remaining = {key.upper(): value for key, value in updates.items()}
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

    ENV_PATH.write_text("\n".join(next_lines).rstrip() + "\n", encoding="utf-8")


def _apply_runtime_env(updates: dict[str, str | int | float | bool | None]) -> None:
    for key, value in updates.items():
        env_key = key.upper()
        if value is None:
            os.environ.pop(env_key, None)
        elif isinstance(value, bool):
            os.environ[env_key] = "true" if value else "false"
        else:
            os.environ[env_key] = str(value)


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
    return hashlib.sha256(_normalize_bridge_base_url(value).encode("utf-8")).hexdigest()


def _bridge_admin_headers(settings=None) -> dict[str, str]:
    settings = settings or get_settings()
    token = settings.model_bridge_admin_token or "local-model-bridge-admin"
    return {"X-Bridge-Admin-Token": token}


def _bridge_base_url(settings=None) -> str:
    settings = settings or get_settings()
    return _normalize_bridge_base_url(settings.chat_base_url or f"http://host.docker.internal:{settings.model_bridge_port}")


def _bridge_target_is_self(value: str | None, settings=None) -> bool:
    normalized = _normalize_bridge_base_url(value)
    if not normalized:
        return False
    settings = settings or get_settings()
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    if host not in {"host.docker.internal", "127.0.0.1", "localhost", "::1", "0.0.0.0"}:
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
        "chat_target_base_url": _normalize_bridge_base_url(env_entries.get("CHAT_BASE_URL") or os.getenv("CHAT_BASE_URL") or ""),
        "chat_resolve_ip": _normalize_bridge_resolve_ip(env_entries.get("CHAT_RESOLVE_IP") or os.getenv("CHAT_RESOLVE_IP") or ""),
        "embedding_target_base_url": _normalize_bridge_base_url(env_entries.get("EMBEDDING_BASE_URL") or os.getenv("EMBEDDING_BASE_URL") or ""),
        "embedding_resolve_ip": _normalize_bridge_resolve_ip(env_entries.get("EMBEDDING_RESOLVE_IP") or os.getenv("EMBEDDING_RESOLVE_IP") or ""),
        "timeout": int(settings.model_request_timeout_seconds or 180),
    }


def model_bridge_status_payload(settings=None, env_entries: dict[str, str] | None = None) -> dict:
    settings = settings or get_settings()
    env_entries = env_entries or _env_entries(ENV_PATH)
    enabled = bool(settings.model_bridge_enabled)
    desired = _desired_bridge_config(settings, env_entries)
    self_target_keys = _bridge_self_target_keys(desired, settings)
    payload: dict = {
        "enabled": enabled,
        "base_url": _bridge_base_url(settings) if enabled else "",
        "desired_chat_target_hash": _hash_bridge_target(str(desired["chat_target_base_url"])),
        "desired_embedding_target_hash": _hash_bridge_target(str(desired["embedding_target_base_url"])),
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
    base_url = _bridge_base_url(settings)
    try:
        health = httpx.get(f"{base_url}/health", timeout=3.0)
        payload["reachable"] = health.status_code == 200
        if health.status_code == 200:
            health_json = health.json()
            payload["config_version"] = health_json.get("config_version")
            payload["chat_target_hash"] = health_json.get("chat_target_hash")
            payload["embedding_target_hash"] = health_json.get("embedding_target_hash")
            payload["routes"] = health_json.get("routes") or {}
    except Exception as exc:
        payload["reachable"] = False
        payload["warnings"].append(f"Model bridge health check failed: {exc}")
        return payload

    try:
        config_response = httpx.get(f"{base_url}/admin/config", headers=_bridge_admin_headers(settings), timeout=3.0)
        payload["admin_available"] = config_response.status_code == 200
        if config_response.status_code != 200:
            payload["warnings"].append(f"Model bridge admin config returned HTTP {config_response.status_code}.")
            return payload
        config = config_response.json()
        payload["config_version"] = config.get("config_version")
        payload["chat_target_hash"] = config.get("chat_target_hash")
        payload["embedding_target_hash"] = config.get("embedding_target_hash")
        payload["config_matches"] = (
            _normalize_bridge_base_url(config.get("chat_target_base_url")) == desired["chat_target_base_url"]
            and _normalize_bridge_base_url(config.get("embedding_target_base_url")) == desired["embedding_target_base_url"]
            and _normalize_bridge_resolve_ip(config.get("chat_resolve_ip")) == desired["chat_resolve_ip"]
            and _normalize_bridge_resolve_ip(config.get("embedding_resolve_ip")) == desired["embedding_resolve_ip"]
            and not self_target_keys
        )
        if not payload["config_matches"]:
            payload["warnings"].append("Model bridge config does not match current .env targets.")
    except Exception as exc:
        payload["admin_available"] = False
        payload["warnings"].append(f"Model bridge admin config failed: {exc}")
    return payload


def reload_model_bridge(settings=None, env_entries: dict[str, str] | None = None) -> dict:
    settings = settings or get_settings()
    env_entries = env_entries or _env_entries(ENV_PATH)
    if not settings.model_bridge_enabled:
        return {"attempted": False, "reason": "model_bridge_disabled"}
    desired = _desired_bridge_config(settings, env_entries)
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
    base_url = _bridge_base_url(settings)
    try:
        response = httpx.post(
            f"{base_url}/admin/reload",
            headers=_bridge_admin_headers(settings),
            json=desired,
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


def current_runtime_settings_version() -> str | None:
    with suppress(Exception):
        version = _redis_client().get(SETTINGS_VERSION_REDIS_KEY)
        if version:
            return str(version)
    return _LAST_RUNTIME_SETTINGS_VERSION


def _version_hash(snapshot: dict, changed_keys: list[str], created_at: str) -> str:
    payload = {
        "settings": snapshot,
        "changed_keys": sorted(set(changed_keys)),
        "created_at": created_at,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _local_runtime_refresh(version: str | None = None) -> None:
    global _LAST_RUNTIME_SETTINGS_VERSION
    get_settings.cache_clear()
    get_settings()
    from app.services.cache_manager import clear_cache_manager

    clear_cache_manager()
    if version is not None:
        _LAST_RUNTIME_SETTINGS_VERSION = version


def publish_runtime_settings_version(changed_keys: list[str], source: str = "api") -> dict:
    snapshot = model_settings_payload()
    snapshot.pop("runtime_settings_version", None)
    snapshot.pop("model_bridge_status", None)
    created_at = datetime.now(timezone.utc).isoformat()
    version_hash = _version_hash(snapshot, changed_keys, created_at)
    message = {
        "version_hash": version_hash,
        "changed_keys": sorted(set(changed_keys)),
        "source": source,
        "created_at": created_at,
    }

    from app.db import SessionLocal
    from app.models import RuntimeSettingsVersion

    with SessionLocal() as session:
        session.add(
            RuntimeSettingsVersion(
                version_hash=version_hash,
                settings_json=snapshot,
                changed_keys_json=message["changed_keys"],
                source=source,
            )
        )
        session.commit()

    client = _redis_client()
    client.set(SETTINGS_VERSION_REDIS_KEY, version_hash)
    client.publish(SETTINGS_VERSION_CHANNEL, json.dumps(message, ensure_ascii=False, sort_keys=True))
    _local_runtime_refresh(version_hash)
    return message


def refresh_runtime_settings_if_needed(force: bool = False) -> dict:
    global _LAST_RUNTIME_SETTINGS_VERSION
    version: str | None = None
    redis_error: str | None = None
    try:
        version = _redis_client().get(SETTINGS_VERSION_REDIS_KEY)
    except Exception as exc:  # pragma: no cover - exercised by integration/runtime checks
        redis_error = str(exc)

    should_refresh = force or (version is not None and version != _LAST_RUNTIME_SETTINGS_VERSION)
    if should_refresh:
        _local_runtime_refresh(version)
    elif version is not None:
        _LAST_RUNTIME_SETTINGS_VERSION = version
    return {
        "refreshed": should_refresh,
        "runtime_settings_version": version or _LAST_RUNTIME_SETTINGS_VERSION,
        "redis_error": redis_error,
    }


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


def normalize_env_file(remove_deprecated: bool = True) -> bool:
    """清理 .env 文件中的 BOM 前缀，不再与 .env.example 合并。

    之前的实现会重写 .env 文件并依赖 .env.example，这会丢失用户注释、
    空行和原有顺序，且 .env.example 是示例文件不应被运行时依赖。
    现在只移除 BOM 前缀，保留文件原貌。
    """
    if not ENV_PATH.exists():
        return False
    content = ENV_PATH.read_text(encoding="utf-8")
    # 移除行首的 BOM 前缀
    cleaned_lines = []
    changed = False
    for line in content.splitlines():
        clean_line = line.lstrip("\ufeff")
        if clean_line != line:
            changed = True
        stripped = clean_line.strip()
        if remove_deprecated and stripped and not stripped.startswith("#") and "=" in clean_line:
            key = clean_line.split("=", 1)[0].strip().upper()
            if key in DEPRECATED_ENV_KEYS:
                changed = True
                continue
        if clean_line.startswith("\ufeff"):
            cleaned_lines.append(clean_line.lstrip("\ufeff"))
            changed = True
        else:
            cleaned_lines.append(clean_line)
    if changed:
        ENV_PATH.write_text("\n".join(cleaned_lines).rstrip() + "\n", encoding="utf-8")
    return changed


def env_sync_status() -> dict:
    """检查 .env 与 .env.example 的参数列表是否一致，并检测 BOM 前缀。"""
    actual_keys, bom_keys = _env_keys(ENV_PATH)
    example_keys, _ = _env_keys(ENV_EXAMPLE_PATH)
    deprecated_keys = sorted(actual_keys & DEPRECATED_ENV_KEYS)
    active_actual_keys = actual_keys - DEPRECATED_ENV_KEYS
    missing_keys = sorted(example_keys - active_actual_keys)
    extra_keys = sorted(active_actual_keys - example_keys)
    return {
        "synced": not bom_keys and not missing_keys and not extra_keys,
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


def _check_model_bridge() -> bool | None:
    settings = get_settings()
    if not settings.model_bridge_enabled:
        return None
    status = model_bridge_status_payload(settings=settings)
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
    infrastructure = {
        "postgres": _check_postgres(),
        "qdrant": _check_qdrant(),
        "redis": _check_redis(),
        "model_bridge": _check_model_bridge(),
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
    bridge_status = model_bridge_status_payload() if get_settings().model_bridge_enabled else {"enabled": False}
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


def update_model_settings(payload: dict) -> dict:
    normalized = normalize_env_file()
    updates: dict[str, str | int | float | bool | None] = {}
    key_map = {
        "chat_base_url": "chat_base_url",
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
        "concept_i18n_enabled": "concept_i18n_enabled",
        "enable_auto_tpe": "enable_auto_tpe",
        "fixed_chunk_size_tokens": "fixed_chunk_size_tokens",
        "fixed_chunk_overlap_tokens": "fixed_chunk_overlap_tokens",
        "context_package_token_budget": "context_package_token_budget",
        "mid_concept_extraction_max_model_batches": "mid_concept_extraction_max_model_batches",
        "mid_concept_extraction_max_candidates_per_batch": "mid_concept_extraction_max_candidates_per_batch",
        "mid_concept_extraction_max_tokens_per_batch": "mid_concept_extraction_max_tokens_per_batch",
        "mid_concept_candidate_keep_threshold": "mid_concept_candidate_keep_threshold",
        "rq_kmeans_levels": "rq_kmeans_levels",
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
        "agent_coarse_total_budget": "agent_coarse_total_budget",
        "agent_mid_per_coarse_budget": "agent_mid_per_coarse_budget",
        "agent_mid_top_k": "agent_mid_top_k",
        "agent_chunk_per_mid_budget": "agent_chunk_per_mid_budget",
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
            updates[env_key] = value.strip() if isinstance(value, str) else value

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

    if updates:
        future_env_entries = dict(_env_entries(ENV_PATH))
        for key, value in updates.items():
            env_key = key.upper()
            if value is None:
                future_env_entries.pop(env_key, None)
            else:
                future_env_entries[env_key] = _serialize_env_value(value)
        future_bridge_enabled = str(future_env_entries.get("MODEL_BRIDGE_ENABLED") or "").strip().lower() in {
            "true",
            "1",
            "yes",
            "on",
        }
        if future_bridge_enabled:
            desired = _desired_bridge_config(get_settings(), future_env_entries)
            self_target_keys = _bridge_self_target_keys(desired, get_settings())
            if self_target_keys:
                raise ValueError(
                    "MODEL_BRIDGE_ENABLED=true 时 CHAT_BASE_URL/EMBEDDING_BASE_URL 必须是真实模型服务地址，不能指向模型桥自身: "
                    + ", ".join(self_target_keys)
                )

    bridge_reload_result: dict | None = None
    if updates:
        _update_env_file(updates)
        _apply_runtime_env(updates)
        get_settings.cache_clear()
        refreshed_settings = get_settings()
        if refreshed_settings.model_bridge_enabled:
            bridge_reload_result = reload_model_bridge(settings=refreshed_settings, env_entries=_env_entries(ENV_PATH))
    if normalized or updates:
        changed_keys = [key.upper() for key in updates]
        if normalized:
            changed_keys.append("ENV_FILE_NORMALIZED")
        from app.services.cache_manager import clear_cache_manager

        clear_cache_manager()
        publish_runtime_settings_version(changed_keys=changed_keys, source="api")
    result = model_settings_payload()
    if bridge_reload_result is not None:
        status = dict(result.get("model_bridge_status") or {})
        status["last_reload"] = bridge_reload_result
        if bridge_reload_result.get("attempted") and not bridge_reload_result.get("ok"):
            status.setdefault("warnings", [])
            status["warnings"].append(f"Model bridge reload failed: {bridge_reload_result.get('error') or bridge_reload_result.get('status_code')}")
        result["model_bridge_status"] = status
    return result
